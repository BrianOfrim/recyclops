import os
import os.path
import logging
import boto3
import botocore
from botocore.exceptions import ClientError

raw_bucket = 'recyclops'
clean_bucket = 'recyclops-clean'

dir_list = ['recycle', 'garbage']
files_to_validate = []

file_type = '.jpg'

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

    # create the output dirctories
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

    for catagory_index, catagory_dir in enumerate(dir_list):
        raw_files = get_files_from_dir(raw_bucket, catagory_dir, file_type)
        clean_files = get_files_from_dir(clean_bucket, catagory_dir, file_type)
        files_to_validate.append([rf for rf in raw_files if not any(cf.key == rf.key for cf in clean_files)])

        logging.info('There are %i items to validate for type %s' % (len(files_to_validate[catagory_index]), catagory_dir))
        download_files(files_to_validate[catagory_index])



if __name__ == '__main__':
    main()
