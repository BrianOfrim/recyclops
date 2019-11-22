import os
import sys
import PySpin
import numpy as np
import cv2
import Jetson.GPIO as GPIO
import time
from os import mkdir
from os.path import isdir, exists
import keyboard
import threading
import queue
import logging
import boto3
from botocore.exceptions import ClientError
from dataclasses import dataclass

WINDOW_NAME = "Recyclops"
FILE_TYPE = "jpg"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BUTTON_INPUT_POLL_SECONDS = 0.01
BUTTON_POLL_COUNT_MAX = 5
CATAGORY_DISPLAY_MILLISECONDS = 1500
DEBOUNCE_POLL_COUNT_MAX = 10
DEBOUNCE_POLL_PERIOD_SECONDS = 0.05
EXTRA_DISPLAY_DELAY_SECONDS = 0.5
NUM_BUFFERS = 10
SEND_TO_CLOUD = True
AWS_S3_BUCKET_NAME = 'recyclops'
s3_client_upload = None
SCALE_FACTOR = 0.5

@dataclass
class Catagory:
    display_name: str
    data_name: str
    button_pin: int
    text_color: tuple
    led_pin: int = 0

@dataclass
class ImageToSave:
    filename: str
    catagory_str: str
    image_data: np.ndarray

recycle_catagory = Catagory('Recycle', 'recycle', 18, (0, 255, 0), 16)
garbage_catagory = Catagory('Garbage', 'garbage', 13, (0, 0, 255), 12)

catagories = [recycle_catagory, garbage_catagory]

image_queue = queue.Queue()

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
        s3_client_upload.upload_file(file_name, bucket, object_name,Callback=ProgressPercentage(file_name))
        print('\n')
    except ClientError as e:
        logging.error(e)
        return False
    return True

def configure_trigger(cam):
    try:
        result = True

        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print('Trigger mode disabled...')

        # Select trigger source
        # The trigger source must be set to hardware or software while trigger
		# mode is off.
        if cam.TriggerSource.GetAccessMode() != PySpin.RW:
            print('Unable to get trigger source (node retrieval). Aborting...')
            return False

        cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)

        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

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

def grab_next_image_by_trigger(cam):
    try:
        result = True

        # Execute software trigger
        if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
            print('Unable to execute trigger. Aborting...')
            return False

        cam.TriggerSoftware.Execute()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True

def debounce_pin(channel):
    readcount = 0
    while(readcount< DEBOUNCE_POLL_COUNT_MAX):
        GPIO.event_detected(channel)
        time.sleep(DEBOUNCE_POLL_PERIOD_SECONDS)
        readcount += 1
	
def process_images():
    while(1):
        image_to_save = image_queue.get(block = True)
        if image_to_save == None:
            break
        filepath = image_to_save.catagory_str + '/' + image_to_save.filename
        print('Image saved at path: %s'% filepath)
        cv2.imwrite(filepath, image_to_save.image_data)
        if SEND_TO_CLOUD:
            upload_file(filepath, AWS_S3_BUCKET_NAME, filepath)
        

def acquire_images(cam):
    """
    This function acquires and saves 10 images from a device.
    Please see Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False
        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

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

        print('Acquiring images. Press enter to end Acquisition.')

        # Get device serial number for filename
        device_serial_number = ''
        if cam.TLDevice.DeviceSerialNumber.GetAccessMode() == PySpin.RO:
            device_serial_number = cam.TLDevice.DeviceSerialNumber.GetValue()

            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve, convert, and save images
        while(keyboard.is_pressed('ENTER') == False):
            try:

                # Retrieve the next image from the trigger
                image_catagory = None
                poll_Count = 0
                while(image_catagory == None and poll_Count < BUTTON_POLL_COUNT_MAX):
                    for catagory in catagories:
                        if GPIO.event_detected(catagory.button_pin):
                            image_catagory = catagory
                    if image_catagory == None:
                        time.sleep(BUTTON_INPUT_POLL_SECONDS)
                        poll_Count += 1

                grab_next_image_by_trigger(cam)

                #  Retrieve next received image
                image_result = cam.GetNextImage()

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    # get a numpy array of the image data
                    imageArray = image_result.GetNDArray()

                    # convert the image from BayerRG8 to RGB8
                    imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BayerRG2RGB)

                    # reduce the image resolution
                    if SCALE_FACTOR != 1:
                        imageArray = cv2.resize(imageArray, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

                    # flip image for mirror effect
                    displayArray = cv2.flip(imageArray, flipCode = 1)


                    displayArray = cv2.putText(displayArray,\
                                   image_catagory.display_name if image_catagory != None else "" ,\
                                   (0,70), FONT, 3,\
                                   image_catagory.text_color if image_catagory != None else None, 3, cv2.LINE_AA)

                    cv2.imshow(WINDOW_NAME, displayArray)

                    cv2.waitKey(1)

                    if(image_catagory != None):
                        if image_catagory.led_pin:
                            print("led pin: %d" % catagory.led_pin)
                            GPIO.output(image_catagory.led_pin, GPIO.HIGH)
                        
                        # Create a unique filename
                        filename = '%s-%d.%s' % (image_catagory.data_name, image_result.GetTimeStamp(), FILE_TYPE)
                        print('Filename: %s, height :%d, width :%d' % (filename, imageArray.shape[0], imageArray.shape[1]))
                        image_queue.put(ImageToSave(filename, image_catagory.data_name, imageArray))
                        debounce_pin(image_catagory.button_pin)
                        time.sleep(EXTRA_DISPLAY_DELAY_SECONDS)
                         
                        if image_catagory.led_pin:
                            GPIO.output(image_catagory.led_pin, GPIO.LOW)
                    
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

    return result


def reset_trigger(cam):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param cam: Camera to acquire images from.
    :type cam: CameraPtr
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Ensure trigger mode off
        # The trigger must be disabled in order to configure whether the source
        # is software or hardware.
        if cam.TriggerMode.GetAccessMode() != PySpin.RW:
            print('Unable to disable trigger mode (node retrieval). Aborting...')
            return False

        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        print('Trigger mode disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
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

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        err = False

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure trigger
        if configure_trigger(cam) is False:
            return False
        
        grab_thread = threading.Thread(target=acquire_images, args = (cam,))
        process_thread = threading.Thread(target=process_images)

        process_thread.start()
        grab_thread.start()

        grab_thread.join()
        print('Finished Acquiring Images')
        image_queue.put(None)
        
        process_thread.join()
        print('Finished Processing Images')

        # Reset trigger
        result &= reset_trigger(cam)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def main():
    result = True

    global SEND_TO_CLOUD

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    
    # button pins set as input
    for catagory in catagories:    
        GPIO.setup(catagory.button_pin, GPIO.IN)
        GPIO.add_event_detect(catagory.button_pin, GPIO.FALLING)
        if catagory.led_pin:
            GPIO.setup(catagory.led_pin, GPIO.OUT)
            GPIO.output(catagory.led_pin, GPIO.LOW)

    # create the output dirctories
    for catagory in catagories:
        if(not isdir(catagory.data_name) or not exists(catagory.data_name)):
            print('Creating output directory: %s' % catagory.data_name)
            try:
                mkdir(catagory.data_name)
            except OSError:
                print ("Creation of the directory %s failed" % catagory.data_name)
                return
            else:
                print ("Successfully created the directory %s " % catagory.data_name)

    
    
    if SEND_TO_CLOUD == True:
        # Check if the bucket exists
        if bucket_exists(AWS_S3_BUCKET_NAME):
            logging.info(f'{AWS_S3_BUCKET_NAME} exists and you have permission to access it.')
        else:
            logging.info(f'{AWS_S3_BUCKET_NAME} does not exist or '
                         f'you do not have permission to access it.')
            SEND_TO_CLOUD = False

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

    GPIO.cleanup()

    input('Done! Press Enter to exit...\n')
    return result


if __name__ == '__main__':
    main()
