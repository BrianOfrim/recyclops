import sys
import os
import os.path
import time
import select
import PySpin
import cv2
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import threading
import queue

WINDOW_NAME = 'Recyclops'
INFERENCE_DEBUG_WINDOW_NAME = 'InferenceDebug'
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
flags.DEFINE_string(
    'inference_model_path',
    '',
    'Path of the tensorflow model to load an use for inference',
)
flags.DEFINE_string(
    'model_storage_dir',
    '../train/savedModels',
    'Direcory where models are saved after training',
)
flags.DEFINE_integer(
    'inference_image_size',
    224,
    'Height and width of the images input into the network',
)
flags.DEFINE_float(
    'confidence_threshold',
    0.8,
    'Confidence above which to display a classification label',
)

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

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

def load_inference_model():
    dir_to_load_from = ''
    labels = []
    if(flags.FLAGS.inference_model_path != ''):
        # check if the model is of the correct type and exists)
        dir_to_load_from = flags.FLAGS.inference_model_path
        if(not os.path.isdir(dir_to_load_from)):
            print('Specified inference model dir %s does not exist' % flags.FLAGS.inference_model_path)
            return None
    else:
        # search for the newest trianed model
        model_storage_dirpath, model_storage_dirs, _ = next(os.walk(flags.FLAGS.model_storage_dir))
        model_storage_dirs.sort(reverse=True)
        if len(model_storage_dirs) == 0:
            print('There are no directories in %s...' % flags.FLAGS.model_storage_dir)
            return None
        dir_to_load_from = os.path.abspath(os.path.join(model_storage_dirpath, model_storage_dirs[0]))
    
    print('File to load model from: %s' % dir_to_load_from)
    with open(dir_to_load_from + '/labels.txt','r') as f:
        for row in f:
            if row.strip('\n'):
                labels.append(row.strip('\n'))
    return (tf.keras.models.load_model(dir_to_load_from), labels)

def process_images(serial_number, image_queue):
    # Fetch the inference model
    classifier, labels = load_inference_model()

    inference_image_height_width = flags.FLAGS.inference_image_size     

    if classifier is not None:
        classifier.build((None, inference_image_height_width, inference_image_height_width, 3))
        classifier.summary()
        cv2.namedWindow(INFERENCE_DEBUG_WINDOW_NAME)
        cv2.moveWindow(INFERENCE_DEBUG_WINDOW_NAME, 0, 0)

    # UI setup
    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    display_scale_factor = flags.FLAGS.display_scale_factor    
    
    while(1):
      
        label_text = ''
        image = image_queue.get(block = True)
        
        if image is None:
            break

        if len(image.shape) < 3:
            # convert the image from BayerRG8 to RGB8
            image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)

        if classifier is not None:
            inference_image = cv2.resize(image, (inference_image_height_width, inference_image_height_width))
            cv2.imshow(INFERENCE_DEBUG_WINDOW_NAME, inference_image)
            inference_image = inference_image/255.0
            result = classifier.predict(inference_image[np.newaxis, ...], verbose=True)
            print(result)
            predicted_class_number = np.argmax(result[0], axis=-1)
            predicted_class_label = labels[predicted_class_number] 
            confidence_level = result[0][predicted_class_number] 
            print("Class: %i, Label: %s ,Confidence: %.4f" % (predicted_class_number, predicted_class_label, confidence_level))            
            if(confidence_level >= flags.FLAGS.confidence_threshold):
                label_text = "%s [%.2f]" % (predicted_class_label, confidence_level)
       
        if flags.FLAGS.mirror_display == True:
            # flip image for mirror effect
            image = cv2.flip(image, flipCode = 1)
       
        if label_text != '':
            image = cv2.putText(image, label_text, (0,80), FONT, 3, (0, 255, 0), 3, cv2.LINE_AA)

        if display_scale_factor != 1:
            # reduce the image resolution
            image = cv2.resize(image, (0,0), fx=display_scale_factor, fy=display_scale_factor)
 
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
        while(1):

            # Check for user input
            if(heardEnter()):
                break

            try:
                while(not triggerReady(cam)):
                    time.sleep(0.001)
                    pass
                
                grab_next_image_by_trigger(cam)
                
                image_result = cam.GetNextImage()

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    # If the process thead is ready, put a numpy array of the image data into the image queue 
                    if image_queue.empty():
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
