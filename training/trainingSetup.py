import logging
import os
import os.path
import re
import boto3
import botocore
from botocore.exceptions import ClientError

raw_bucket = 'recyclops'
clean_dir = 'verified'
catagory_dir_list = ['recycle', 'garbage']

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

def get_current_verification_list(bucket_name, verified_list_dir, verification_type, catagory):
    files_from_dir = get_files_from_dir(bucket_name, verified_list_dir + "/" + catagory +"/" +  verification_type, ".txt")
    if len(files_from_dir) == 0: return None
    #find the most recent (highest timestamp)
    sorted_files = sorted(files_from_dir, key = lambda summary: int(re.findall('[0-9]+', summary.key)[0]), reverse=True)
    #download the most recent verification file
    download_files(bucket_name, [sorted_files[0].key])
    with open(sorted_files[0].key, "r") as f:
        return [line.strip() for line in f if line.strip()] 
    return []

def download_files(bucket_name, file_names):
    s3 = boto3.client('s3')
    for object_index, object_name in enumerate(file_names):
        if(not os.path.isfile(object_name)):
            try:
                s3.download_file(bucket_name, object_name, object_name)
            except botocore.exceptions.ClientError as e:
                logging.error(e)
        logging.info('Downloading file from %s:%s, %i/%i' % \
            (bucket_name, object_name, object_index + 1, len(file_names)))

def main():
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

    # Create the classification directories
    create_output_dir(clean_dir)
    for catagory_dir in catagory_dir_list:
        create_output_dir(catagory_dir)
        create_output_dir(clean_dir + '/' + catagory_dir)
        create_output_dir(clean_dir + '/' + catagory_dir + '/valid')

    # Get the most recent verified file list
    verified_files = dict((c,  []) for c in catagory_dir_list)
    for catagory_dir in catagory_dir_list:
        verified_files[catagory_dir] = get_current_verification_list(raw_bucket, clean_dir, "valid", catagory_dir)
    
    # Fetch the images
    for catagory_dir in catagory_dir_list:
        download_files(raw_bucket, verified_files[catagory_dir])

if __name__ == '__main__':
    main()
