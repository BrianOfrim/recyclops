import os
import os.path
from sys import exit
import logging
import time
import shutil
import re
import numpy as np
import cv2
import boto3
import botocore
from botocore.exceptions import ClientError
from signal import signal, SIGINT
from absl import app
from absl import flags

WINDOW_NAME = "Recyclops"
raw_bucket = 'recyclops'    
verified_file_dir = 'verified'
file_type = '.jpg'
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR_DISPLAY = (81, 237, 14)
FONT_COLOR_CATEGORY = (237, 181, 14)

flags.DEFINE_spaceseplist(
    'input_categories_list',
    'aluminum compost glass paper plastic trash',
    'List of catoagories to clean files from',
)
flags.DEFINE_spaceseplist(
    'output_categories_list',
    'aluminum compost glass paper plastic trash invalid',
    'List of categories to add cleaned files to',
)

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

def file_exists(bucket_name, object_path):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket_name, object_path).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True

def upload_files(bucket, files_to_send, check_if_file_exists = True):
    s3 = boto3.client('s3')
    for file_index, file_to_send in enumerate(files_to_send):
        try:
            if(check_if_file_exists and file_exists(bucket, file_to_send)):
                logging.info("File already exists %s:%s, %i/%i" % \
                    (bucket, file_to_send, file_index + 1, len(files_to_send)))
                continue
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
                s3.download_file(object_summary.bucket_name, object_summary.key, 
                    object_summary.key)
            except botocore.exceptions.ClientError as e:
                logging.error(e)
            logging.info('Downloading file from %s:%s, %i/%i' % \
                (object_summary.bucket_name, object_summary.key, object_index + 1, \
                    len(object_summary_list)))
        else:
             logging.info('File already downloaded: %s:%s, %i/%i' % \
                (object_summary.bucket_name, object_summary.key, object_index + 1, \
                    len(object_summary_list)))

def get_current_verification_list(bucket_name, category, verification_dir):
    files_from_dir = get_files_from_dir(bucket_name, category  + '/' +  verification_dir, ".txt")
    if len(files_from_dir) == 0: return None
    #find the most recent (highest timestamp)
    sorted_files = sorted(files_from_dir, key = lambda summary: int(re.findall('[0-9]+', \
        summary.key)[0]), reverse=True)
    #download the most recent verification file
    download_files([sorted_files[0]])
    with open(sorted_files[0].key, "r") as f:
        return [line.strip() for line in f if line.strip()] 
    return None
    
def upload_verification_file(bucket_name, category, verification_dir, \
        upload_time, image_file_names):
    if(len(image_file_names) == 0): return
    # create a file containing the verified files
    verification_file_name = '%s/%s/%i-%s.txt' % \
        (category, verification_dir, upload_time, category)
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

def main(unused_argv):
    
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

    input_category_dir_list = flags.FLAGS.input_categories_list

    if(len(input_category_dir_list) == 0):
        print('No input directories specified')
        exit(0)

    print('Input categories:')
    print(input_category_dir_list)
 
    output_category_dir_list = flags.FLAGS.output_categories_list 

    if(len(output_category_dir_list) == 0):
        print('No output directories specified')
        exit(0)
    
    print('Output categories:')
    print(output_category_dir_list)

      
    # Create the output dirctories
    for output_category_dir in output_category_dir_list:
        create_output_dir(output_category_dir)
        create_output_dir(output_category_dir + '/' + verified_file_dir)

    for input_category_dir in input_category_dir_list:
        create_output_dir(input_category_dir)
        
    already_validated_files = set()   

    # Get the most recent verified file list for input categories
    for input_category_dir in  input_category_dir_list:
        verification_list = get_current_verification_list(raw_bucket, input_category_dir,\
                                verified_file_dir)
        if(verification_list):
            already_validated_files.update([vf.split('/', 1)[1] for vf in verification_list])
 
    # Get the most recent verified file list for output categories
    original_output_lists = dict((c, set()) for c in output_category_dir_list)
    for output_category_dir in output_category_dir_list:
        verification_list = get_current_verification_list(raw_bucket, output_category_dir,\
                                verified_file_dir)
        if(verification_list):
            original_output_lists[output_category_dir] = set(verification_list)
            already_validated_files.update([vf.split('/', 1)[1] for vf in verification_list])
    
    # Create a dictionay of lists to store newly vailidated file names
    current_output_lists = dict((c, set()) for c in output_category_dir_list)
    
    files_to_validate = dict((c, []) for c in input_category_dir_list)

    # Fetch the images
    for input_category_dir in input_category_dir_list:
        input_files = get_files_from_dir(raw_bucket, input_category_dir, file_type) 
        
        files_to_validate[input_category_dir].extend(\
            [inF for inF in input_files if inF.key.split('/', 1)[1]\
                not in already_validated_files])
        logging.info('There are %i items to sort for type %s' % \
            (len(files_to_validate[input_category_dir]), input_category_dir))
        download_files(list(files_to_validate[input_category_dir]))

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    
    continue_display = True

    info_str = ''
    for dir_index, output_category_dir in enumerate(output_category_dir_list):
        info_str += "%s:['%i'] " % (output_category_dir, dir_index + 1)
 
    # Show the data to the user
    for input_category_dir in input_category_dir_list:
        file_index = 0
        while file_index < len(files_to_validate[input_category_dir]):
            logging.info("Category: %s Image: %i/%i  Filename: %s " % \
                    (input_category_dir, file_index, len(files_to_validate[input_category_dir]),\
                    files_to_validate[input_category_dir][file_index]))
            current_image = cv2.imread(files_to_validate[input_category_dir][file_index].key)

            display_image = np.copy(current_image)
            # display input options
            display_image = cv2.putText(display_image, info_str, (0,30), FONT_TYPE,\
                 0.7, FONT_COLOR_DISPLAY, 1, cv2.LINE_AA)
            display_image = cv2.putText(display_image, 'Current classification: %s' % input_category_dir, (0,60), FONT_TYPE,\
                 0.7, FONT_COLOR_CATEGORY, 1, cv2.LINE_AA)

            # display current clasification
            cv2.imshow(WINDOW_NAME, display_image)
            keypress = cv2.waitKey(0)

           # Get the user input
            if(keypress == 27):
                # escape key pressed move to next category
                break
            elif(keypress & 0xFF == ord('w')):
                # return to previous
                if(file_index > 0):
                    print("Previous.")
                    file_index -= 1
                else:
                    print("Already at the start")
            elif(keypress & 0xFF == ord('s')):
                #skip
                print("Skip")
                for key in current_output_lists:
                    current_output_lists[key].discard(\
                        files_to_validate[input_category_dir][file_index].key)
                file_index += 1
            for dir_index, output_category_dir in enumerate(output_category_dir_list):
                if(keypress & 0xFF == ord(str(dir_index + 1))):
                    print("Sorted into: %s" % output_category_dir)
                    # add to the choosen list
                    current_output_lists[output_category_dir].add(\
                        files_to_validate[input_category_dir][file_index].key)
                    # discard from all other lists
                    for key in current_output_lists:
                        if(key != output_category_dir):
                            current_output_lists[key].discard(\
                                files_to_validate[input_category_dir][file_index].key)
                    category_image = np.copy(current_image) 
                    category_image = cv2.putText(category_image, "%s:['%i']" % \
                        (output_category_dir, dir_index + 1) \
                        , (0,30), FONT_TYPE, 0.7, FONT_COLOR_CATEGORY, 1, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, category_image)
                    cv2.waitKey(300)
                    file_index +=1
                    break
            
    cv2.destroyAllWindows()

    upload_time = time.time()
    
    # print the result
    for category_name in current_output_lists:
        output_file_names = []
        # copy files from input dir to output dir
        for input_file_name in current_output_lists[category_name]:
            output_file_name = category_name + '/' + input_file_name.split('/', 1)[1]
            if(not os.path.isfile(output_file_name)):
                shutil.copyfile(input_file_name, output_file_name)
            output_file_names.append(output_file_name)
       
        # upload new files 
        upload_files(raw_bucket, output_file_names)
        if(len(output_file_names) != 0):
            upload_verification_file(raw_bucket, category_name, verified_file_dir,\
                upload_time, original_output_lists[category_name].union(set(output_file_names)))

if __name__ == "__main__":
  app.run(main)
