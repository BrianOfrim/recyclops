import os.path
import time
import keyboard
import PySpin
import cv2
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import threading
import queue

WINDOW_NAME = 'Recyclops'
FONT = cv2.FONT_HERSHEY_SIMPLEX
NUM_BUFFERS = 3
TRIGGER_READY_LINE = 'Line2'

flags.DEFINE_float(
    'display_scale_factor',
    1,
    'Factor to scale images by for display ',
)
flags.DEFINE_bool(
    'mirror_display',
    True,
    'Mirror the images for display',
)

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
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    display_scale_factor = flags.FLAGS.display_scale_factor    
    while(1):
        image = image_queue.get(block = True)
        if image is None:
            break
        if len(image.shape) < 3:
            # convert the image from BayerRG8 to RGB8
            image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        
        if display_scale_factor != 1:
            # reduce the image resolution
            image = cv2.resize(image, (0,0), fx=display_scale_factor, fy=display_scale_factor)
        
        if flags.FLAGS.mirror_display:
            # flip image for mirror effect
            image = cv2.flip(image, flipCode = 1)
        
        cv2.imshow(WINDOW_NAME, image)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()


def acquire_images(cam, image_queue):
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

        print('Acquiring images. Press enter to end Acquisition.')

        # Get device serial number for filename
        device_serial_number = cam.GetUniqueID()

        # Retrieve, convert, and save images
        while(keyboard.is_pressed('ENTER') == False):
            try:
                while(not triggerReady(cam)):
                    time.sleep(0.001)
                    pass
                
                grab_next_image_by_trigger(cam)
                
                image_result = cam.GetNextImage()

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    # Put a numpy array of the image data into the image queue 
                    image_queue.put(image_result.GetNDArray())
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

def main(unused_argv):
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

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

    # Grab the first available camera
    cam = cam_list.GetByIndex(0)
    run_single_camera(cam)
    del cam
    
    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()
    
    print('Exititng ...\n') 


if __name__ == "__main__":
  app.run(main)
