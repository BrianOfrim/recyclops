import os
import os.path
from sys import exit
import logging
import time
import re
import numpy as np
import cv2
import boto3
import botocore
from botocore.exceptions import ClientError
from signal import signal, SIGINT

WINDOW_NAME = "Recyclops"
raw_bucket = 'recyclops'
clean_dir = 'verified'
input_catagory_dir = ['recycle']
output_catagory_dir_list = [('paper','g'), ('aluminumContainer','h'), ('glassContainer', 'j'), ('plasticContainer','k') , ('invalid', 'l')]
file_type = '.jpg'
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR_DISPLAY = (0, 255, 0)

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
    files_from_dir = []
    for object_summary in my_bucket.objects.filter(Prefix=dir_name + '/'):
        if(object_summary.key.endswith(file_extension)):
            files_from_dir.append(object_summary)
    return files_from_dir 

   

def upload_files(bucket, files_to_send):
    s3 = boto3.client('s3')
    for file_index, file_to_send in enumerate(files_to_send):
        try:
            s3.upload_file(file_to_send, bucket, file_to_send)
        except ClientError as e:
            logging.error(e)
        logging.info("Uploading file to %s:%s, %i/%i" % \
            (bucket, file_to_send, file_index + 1, len(files_to_send)))

def download_files(object_summary_list):
    s3 = boto3.client('s3')
    for object_index, object_summary in enumerate(object_summary_list):
        if(not os.path.isfile(object_summary.key)):
            try:
                s3.download_file(object_summary.bucket_name, object_summary.key, object_summary.key)
            except botocore.exceptions.ClientError as e:
                logging.error(e)
        logging.info('Downloading file from %s:%s, %i/%i' % \
            (object_summary.bucket_name, object_summary.key, object_index + 1, len(object_summary_list)))


def get_current_verification_list(bucket_name, catagory):
    files_from_dir = get_files_from_dir(catagory, ".txt")
    if len(files_from_dir) == 0: return None
    #find the most recent (highest timestamp)
    sorted_files = sorted(files_from_dir, key = lambda summary: int(re.findall('[0-9]+', summary.key)[0]), reverse=True)
    #download the most recent verification file
    download_files([sorted_files[0]])
    with open(sorted_files[0].key, "r") as f:
        return [line.strip() for line in f if line.strip()] 
    return None
    
def upload_verification_file(bucket_name, verification_dir ,catagory, verification_type, image_file_names):
    if(len(image_file_names) == 0): return        
    # create a file containing the verified files
    verification_file_name = '%s/%s/%s/%i-%s-%s.txt' % \
        (verification_dir, catagory, verification_type, time.time(), catagory, verification_type)
    with open(verification_file_name, 'w') as f:
        for fn in image_file_names:
            f.write("%s\n" % fn) 
    upload_files(bucket_name, [verification_file_name])


def create_output_dir(dir_name):
    if(not os.path.isdir(dir_name) or not os.path.exists(dir_name)):
        print('Creating output directory: %s' % dir_name)
        try:
            os.mkdir(dir_name)
        except OSError:
            print ("Creation of the directory %s failed" % dir_name)
            return
        else:
            print ("Successfully created the directory %s " % dir_name)


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
       
    # Create the output dirctories
    create_output_dir(clean_dir)
    for output_catagory_dir,_ in output_catagory_dir_list:
        create_output_dir(output_catagory_dir)

    # Get the most recent verified file list
    current_output_lists = dict((c, set()) for c, _ in output_catagory_dir_list)
    for output_catagory_dir in output_catagory_dir_list:
        verification_list = get_current_verification_list(raw_bucket, output_catagory_dir)
        if(verification_list):
            current_output_lists[output_catagory_dir] = set(verification_list)

    files_to_validate = dict((c, []) for c in input_catagory_dir_list)

    # Fetch the images
    for input_catagory_dir in input_catagory_dir_list:
        input_files = get_files_from_dir(raw_bucket, input_catagory_dir, file_type) 
        clean_files = set()
        for output_catagory_diri, _ in output_catagory_dir_list:
            clean_files.update(current_output_lists[output_catagory_dir])
        files_to_validate[input_catagory_dir].extend([inF for inF in input_files if inF.key not in clean_files])
        logging.info('There are %i items to sort for type %s' % (len(files_to_validate[input_catagory_dir]), input_catagory_dir))
        download_files(list(files_to_validate[input_catagory_dir]))

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    
    continue_display = True

    # Show the data to the user
    for input_catagory_dir in input_catagory_dir_list:
        file_index = 0
        while file_index < len(files_to_validate[input_catagory_dir]):
            logging.info("Catagory: %s Image: %i/%i  Filename: %s " % \
                    (input_catagory_dir, file_index, len(files_to_validate[input_catagory_dir]),\
                    files_to_validate[input_catagory_dir][file_index]))
            current_image = cv2.imread(files_to_validate[input_catagory_dir][file_index].key)
            
            display_image = cv2.putText(current_image, input_catagory_dir, (0,70), FONT_TYPE, 3, FONT_COLOR_DISPLAY, 3, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display_image)
            keypress = cv2.waitKey(0)

            if(keypress == 27):
                # escape key pressed
                break

           # Get the user input
            if(keypress & 0xFF == ord('w')):
                # return to previous
                if(file_index > 0):
                    print("Previous.")
                    file_index -= 1
                else:
                    print("Already at the start")
                continue

            elif(keypress & 0xFF == ord('s')):
                #skip
                print("Skip")
                for key in current_output_listis:
                    current_output_lists[key].discard(files_to_validate[input_catagory_dir][file_index].key)
                file_index += 1
                continue

            for output_catagory_dir, catagory_char in output_catagory_dir_list:
                if(keypress & 0xFF == ord(catagory_char)):
                    print("Sorted into: %s" % output_catagory_dir)
                    # add to the choosen list
                    current_output_lists[output_catagory_dir].add(files_to_validate[input_catagory_dir][file_index].key)
                    # discard from all other lists
                    for key in current_output_listis:
                        if(key != output_catagory_dir):
                            current_output_lists[key].discard(files_to_validate[input_catagory_dir][file_index].key)
                    display_image = cv2.putText(current_image, "%s ['%s']" % (output_catagory_dir, catagory_char) \
                        , (0,70), FONT_TYPE, 3, FONT_COLOR_DISPLAY, 3, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, display_image)
                    cv2.waitKey(300)

            #invalid input
            print("Invalid input, please try again")

            
    cv2.destroyAllWindows()

    # print the results
    for key in current_output_lists:
        for item in current_output_lists[key]:
            print('List: %s, Item %s' % (key, item))

    #for catagory_dir in outut_catagory_dir_list:
    #        upload_verification_file(raw_bucket, clean_dir, catagory_dir, verification_type,\
    #             current_verification_lists[catagory_dir][verification_type])


    
if __name__ == '__main__':
    main()
