import os
import sys
import PySpin
from absl import app
from absl import flags
import numpy as np
import cv2
import time
from os import mkdir
from os.path import isdir, exists
import threading
import queue
import logging
import boto3
from botocore.exceptions import ClientError
from dataclasses import dataclass

WINDOW_NAME = "Recyclops"
FONT = cv2.FONT_HERSHEY_SIMPLEX
CATEGORY_DISPLAY_MILLISECONDS = 750
NUM_BUFFERS = 3
INFO_COLOR = (81, 237, 14)

s3_client_upload = None

flags.DEFINE_string(
    'image_file_type',
    'jpg',
    'File format to saving images as',
)
flags.DEFINE_bool(
    'send_to_cloud',
    True,
    'Should captured images be sent to S3?',
)
flags.DEFINE_string(
    's3_bucket_name',
    'recyclops',
    'Name of the s3 bucket to send images to'
)
flags.DEFINE_float(
    'display_scale_factor',
    0.5,
    'Scale factor to apply to displayed images',
)
flags.DEFINE_float(
    'save_scale_factor',
    0.5,
    'Scale factor to apply to saved images',
)
flags.DEFINE_bool(
    'mirror_display',
    True,
    'Mirror the displayed image',
)

@dataclass
class Category:
    display_name: str
    data_name: str
    text_color: tuple
    keyboard_string: str

@dataclass
class ImageToSave:
    filename: str
    category_str: str
    image_data: np.ndarray

aluminum_category = Category('Aluminum', 'aluminum', (237, 181, 14), '1')
compost_category = Category('Compost', 'compost', (219, 56, 210), '2')
glass_category = Category('Glass', 'glass', (255, 74, 164), '3')
paper_category = Category('Paper', 'paper', (230, 245, 24), '4')
plastic_category = Category('Plastic', 'plastic', (24, 230, 245), '5')
trash_category = Category('Trash', 'trash', (24, 171, 245), '0')


categories = [aluminum_category, compost_category, glass_category,\
                paper_category, plastic_category, trash_category]

def bucket_exists(bucket_name):
    """Determine whether bucket_name exists and the user has permission to access it

    :param bucket_name: string
    :return: True if the referenced bucket_name exists, otherwise False
    """

    s3_client_bucket_check = boto3.client('s3')
    try:
        response = s3_client_bucket_check.head_bucket(Bucket=bucket_name)
    except Exception as e:
        logging.debug(e)
        return False
    return True

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def upload_file(file_name, bucket, object_name=None):

    global s3_client_upload
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    if s3_client_upload == None:
        s3_client_upload = boto3.client('s3')
    try:
        s3_client_upload.upload_file(file_name, bucket, object_name,
                Callback=ProgressPercentage(file_name))
        print('\n')
    except ClientError as e:
        logging.error(e)
        return False
    return True

def print_device_info(nodemap):
    print('*** DEVICE INFORMATION ***\n')
    try:
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
        else:
            print('Device control information not available.')
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def configure_trigger(cam):
    try:
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def reset_trigger(cam):
    try:
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def configure_trigger_ready_line(cam):
    try:
        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        cam.LineMode.SetValue(PySpin.LineMode_Output)
        lineSourceNode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode("LineSource"))
        frameTriggerWaitEntry = lineSourceNode.GetEntryByName("FrameTriggerWait") 
        lineSourceNode.SetIntValue(frameTriggerWaitEntry.GetValue())
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def triggerReady(cam):
    return (cam.LineStatusAll() & (1 << 2)) != 0

def grab_next_image_by_trigger(cam):
    try:
        # Execute software trigger
        if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
            print('Unable to execute trigger. Aborting...')
            return False
        cam.TriggerSoftware.Execute()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def process_images(serial_number, image_queue):
    while(1):
        image_to_save = image_queue.get(block = True)
        if image_to_save == None:
            break
        filepath = image_to_save.category_str + '/' + image_to_save.filename
        print('Image saved at path: %s'% filepath)
        cv2.imwrite(filepath, image_to_save.image_data)
        if flags.FLAGS.send_to_cloud:
            upload_file(filepath, flags.FLAGS.s3_bucket_name, filepath)

def acquire_images(cam, image_queue):
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    try:
        # Stop Acquisition if image is streaming
        if(cam.IsStreaming()):
            cam.EndAcquisition()

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))

        buffer_count.SetValue(NUM_BUFFERS)

        # Display Buffer Info
        print('Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        print('Buffer Count: %d' % buffer_count.GetValue())
        print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        buffer_count.SetValue(NUM_BUFFERS)

        # Set acquisition mode to continuous
        if cam.AcquisitionMode.GetAccessMode() != PySpin.RW:
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False

        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images. Press esc to end Acquisition.')

        # Get device serial number for filename
        device_serial_number = cam.GetUniqueID()
        
        info_string = ''
        for category in categories:
            info_string += "%s:'%s' " % (category.display_name, category.keyboard_string)

        # Retrieve, convert, and save images
        while(1):
            try:
                while(not triggerReady(cam)):
                    time.sleep(0.001)
                    pass
                
                grab_next_image_by_trigger(cam)

                #  Retrieve next received image
                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    # get a numpy array of the image data
                    imageArray = image_result.GetNDArray()

                    if len(imageArray.shape) < 3:
                        # convert the image from BayerRG8 to RGB8
                        imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BayerRG2RGB)

                    displayArray = np.copy(imageArray)

                    if flags.FLAGS.mirror_display:
                         displayArray = cv2.flip(displayArray, flipCode = 1)

                    displayArray = cv2.putText(displayArray,\
                                    info_string, (0,50), FONT, 2,\
                                    INFO_COLOR, 2, cv2.LINE_AA)
                        
                    if flags.FLAGS.display_scale_factor != 1:
                        displayArray = cv2.resize(displayArray, (0,0), 
                                fx=flags.FLAGS.display_scale_factor,
                                fy=flags.FLAGS.display_scale_factor)

                    cv2.imshow(WINDOW_NAME, displayArray)
                    keypress = cv2.waitKey(1)

                    if keypress == 27:
                        # escape key pressed
                        break

                    image_category = None

                    for category in categories:
                        if(keypress & 0xFF == ord(category.keyboard_string)):
                            image_category = category

                    if(image_category != None):
                        # Create a unique filename
                        filename = '%s-%d.%s' % (image_category.data_name,
                                image_result.GetTimeStamp(), flags.FLAGS.image_file_type)
                        print('Filename: %s, height :%d, width :%d' % 
                                (filename, imageArray.shape[0], imageArray.shape[1]))

                        saveArray = np.copy(imageArray)

                        if flags.FLAGS.save_scale_factor != 1:
                            saveArray = cv2.resize(imageArray, (0,0), 
                                fx=flags.FLAGS.save_scale_factor,
                                fy=flags.FLAGS.save_scale_factor)

                        image_queue.put(ImageToSave(filename, image_category.data_name, saveArray))

                        displayArray = np.copy(imageArray)

                        if flags.FLAGS.mirror_display:
                             displayArray = cv2.flip(displayArray, flipCode=1)

                        displayArray = cv2.putText(displayArray,\
                                    image_category.display_name , (0,50), FONT, 2,\
                                    image_category.text_color, 2, cv2.LINE_AA)

                        if flags.FLAGS.display_scale_factor != 1:
                            displayArray = cv2.resize(displayArray, (0,0), 
                                fx=flags.FLAGS.display_scale_factor,
                                fy=flags.FLAGS.display_scale_factor)
                        
                        cv2.imshow(WINDOW_NAME, displayArray)
                        cv2.waitKey(CATEGORY_DISPLAY_MILLISECONDS)
                        
                    # Release image
                    image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        # End acquisition
        cam.EndAcquisition()
        cv2.destroyAllWindows()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):
    try:
        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure trigger
        if configure_trigger(cam) is False:
            return False

        # Configure trigger ready line
        if configure_trigger_ready_line(cam) is False:
            return False

        image_queue = queue.Queue()

        grab_thread = threading.Thread(target=acquire_images, args=(cam, image_queue,))
        process_thread = threading.Thread(target=process_images, args=(cam.GetUniqueID(), image_queue,))

        process_thread.start()
        grab_thread.start()

        grab_thread.join()
       
        print('Finished Acquiring Images')
        image_queue.put(None)
        
        process_thread.join()
        print('Finished Processing Images')
        
        # Reset trigger
        reset_trigger(cam)
        # Deinitialize camera
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False
    return True

def main(unused_argv):
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    
    # create the output dirctories
    for category in categories:
        if(not isdir(category.data_name) or not exists(category.data_name)):
            print('Creating output directory: %s' % category.data_name)
            try:
                mkdir(category.data_name)
            except OSError:
                print ("Creation of the directory %s failed" % category.data_name)
                return
            else:
                print ("Successfully created the directory %s " % category.data_name)
    
    if flags.FLAGS.send_to_cloud:
        # Check if the bucket exists
        if bucket_exists(flags.FLAGS.s3_bucket_name):
            print('%s exists and you have permission to access it.' % flags.FLAGS.s3_bucket_name)
        else:
            print('%s does not exist or you do not have permission to access it.' % flags.FLAGS.s3_bucket_name)
            return

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    cam = cam_list.GetByIndex(0)
    print('Running example for camera...')
    
    run_single_camera(cam)
    print('Camera example complete... \n')

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    print('Exiting...\n')

if __name__ == "__main__":
  app.run(main)
