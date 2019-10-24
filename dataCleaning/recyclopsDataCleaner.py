import os
import os.path
import logging
import numpy as np
import cv2
import boto3
import botocore
from botocore.exceptions import ClientError
from sys import exit
from signal import signal, SIGINT

WINDOW_NAME = "Recyclops"
raw_bucket = 'recyclops'
clean_bucket = 'recyclops-clean'
dir_list = ['recycle', 'garbage']
file_type = '.jpg'
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR_DISPLAY = (255, 0, 0)
FONT_COLOR_VALID = (0, 255, 0)
FONT_COLOR_INVALID = (0, 0, 255)


def exit_handler(signal_received, frame):
    print("Forced exit...")
    cv2.destroyAllWindows()
    exit(0)

def bucket_exists(bucket_name):
    """Determine whether bucket_name exists and the user has permission to access it
    :param bucket_name: string
    :return: True if the referenced bucket_name exists, otherwise False
    """

    s3 = boto3.client('s3')
    try:
        response = s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        logging.debug(e)
        return False
    return True


def get_files_from_dir(bucket_name, dir_name, file_extension):

    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)

    valid_files = []
    
    for object_summary in my_bucket.objects.filter(Prefix=dir_name + '/'):
        if(object_summary.key.endswith(file_extension)):
            valid_files.append(object_summary)
    return valid_files    

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
        s3_client_upload.upload_file(file_name, bucket, object_name)
        print('\n')
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_files(object_summary_list):
    
    s3 = boto3.client('s3')
    for object_index, object_summary in enumerate(object_summary_list):
        if(not os.path.isfile(object_summary.key)):
            try:
                s3.download_file(object_summary.bucket_name, object_summary.key, object_summary.key)
            except botocore.exceptions.ClientError as e:
                logging.error(e)
        logging.info('Downloading file from %s:%s, %i/%i' % \
            (object_summary.bucket_name, object_summary.key, object_index, len(object_summary_list)))

def main():

    signal(SIGINT, exit_handler)

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    # Check if the raw bucket exists
    if bucket_exists(raw_bucket):
        logging.info(f'{raw_bucket} exists and you have permission to access it.')
    else:
        logging.info(f'{raw_bucket} does not exist or '
                     f'you do not have permission to access it.')
        return

    # Check if the clean bucket exists
    if bucket_exists(clean_bucket):
        logging.info(f'{clean_bucket} exists and you have permission to access it.')
    else:
        logging.info(f'{clean_bucket} does not exist or '
                     f'you do not have permission to access it.')
        return

    # Create the output dirctories
    for catagory_dir in dir_list:
        if(not os.path.isdir(catagory_dir) or not os.path.exists(catagory_dir)):
            print('Creating output directory: %s' % catagory_dir)
            try:
                os.mkdir(catagory_dir)
            except OSError:
                print ("Creation of the directory %s failed" % catagory_dir)
                return
            else:
                print ("Successfully created the directory %s " % catagory_dir)


    files_to_validate = []

    # Fetch the images
    for catagory_index, catagory_dir in enumerate(dir_list):
        raw_files = get_files_from_dir(raw_bucket, catagory_dir, file_type)
        clean_files = get_files_from_dir(clean_bucket, catagory_dir, file_type)
        files_to_validate.append([rf for rf in raw_files if not any(cf.key == rf.key for cf in clean_files)])

        logging.info('There are %i items to validate for type %s' % (len(files_to_validate[catagory_index]), catagory_dir))
        download_files(files_to_validate[catagory_index])

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    
    
    
    valid_files = {}
    invalid_files = {}
    continue_display = True

    # Show the data to the user
    for catagory_index, catagory_dir in enumerate(dir_list):
        file_index = 0
        while file_index < len(files_to_validate[catagory_index]):
            logging.info("Catagory: %s Image: %i/%i  Filename: %s " % \
                    (catagory_dir, file_index, len(files_to_validate[catagory_index]),\
                    files_to_validate[catagory_index][file_index]))
            current_image = cv2.imread(files_to_validate[catagory_index][file_index].key)
            
            display_image = cv2.putText(current_image, catagory_dir, (0,70), FONT_TYPE, 3, FONT_COLOR_DISPLAY, 3, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display_image)
            keypress = cv2.waitKey(0)
            # Get the user input
            if(keypress & 0xFF == ord('d')):
                # image is valid
                print("Valid!")
                display_image = cv2.putText(current_image, catagory_dir + " valid" \
                    , (0,70), FONT_TYPE, 3, FONT_COLOR_VALID, 3, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display_image)
                cv2.waitKey(300)
                files_to_validate[catagory_index][file_index]
                print(files_to_validate[catagory_index][file_index])
                file_index += 1
            elif(keypress & 0xFF == ord('a')):
                #image is invalid
                print("Invalid...")
                display_image = cv2.putText(current_image, catagory_dir + " invalid" \
                    , (0,70), FONT_TYPE, 3, FONT_COLOR_INVALID, 3, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display_image)
                cv2.waitKey(300)
                files_to_validate[catagory_index][file_index].valid = False
                file_index += 1 
            elif(keypress & 0xFF == ord('w')):
                # return to previous
                if(file_index > 0):
                    print("Previous.")
                    file_index -= 1
                else:
                    print("Already at the start")
            elif(keypress & 0xFF == ord('p')):
                # exit
                continue_display = False
                print("Exit cleaning...")
                break
            else:
                #invalid input
                print("Invalid input, please try again")
        
        print("Continue display %r" % continue_display)
        if(continue_display == False):
            break

    cv2.destroyAllWindows()

    file_count = 0

    # Count files to upload
#    for catagory_index, catagory_dir in enumerate(dir_list):
#        file_count += sum(f.valid for f in files_to_validate[catagory_index])

    # Upload clean 
    for catagory_index, catagory_dir in enumerate(dir_list):
        for file_index, file_summary in enumerate(files_to_validate[catagory_index]):        
        #    if(files_to_validate[catagory_index][file_index].valid):
            print(files_to_validate[catagory_index][file_index])                

if __name__ == '__main__':
    main()
