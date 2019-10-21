import boto3

raw_bucket = 'recyclops'
clean_bucket = 'recyclops-clean'

dir_list = ['recycle', 'garbage']
files_to_validate = []


file_type = '.jpg'

import logging
import boto3
from botocore.exceptions import ClientError


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


    for catagory_index, catagory_dir in enumerate(dir_list):
        raw_files = get_files_from_dir(raw_bucket, catagory_dir, file_type)
        clean_files = get_files_from_dir(clean_bucket, catagory_dir, file_type)
        files_to_validate.append([rf for rf in raw_files if not any(cf.key == rf.key for cf in clean_files)])

        logging.info('There are %i items to validate for type %s' % (len(files_to_validate[catagory_index]), catagory_dir))
        



if __name__ == '__main__':
    main()
