import os
import os.path
import sys
import time
import shutil
import threading
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import boto3
from botocore.exceptions import ClientError

s3_bucket_name = 'recyclops'

flags.DEFINE_string(
    'input_dir',
    './images',
    'The directory where the input images are stored',
)
flags.DEFINE_integer(
    'num_epochs',
    10,
    'Number of epochs to train the model for',
)
flags.DEFINE_integer(
    'batch_size',
    8,
    'Number of examples in a batch',
)
flags.DEFINE_float(
    'dropout',
    0.0,
    'Dropout rate for layer before gloal pooling',
)
flags.DEFINE_float(
    'val_split',
    0.2,
    'The percentage of input data to use for valuation',
)
flags.DEFINE_bool(
    'fine_tune',
    False,
    'Unfreeze lower layers of model to fine tune'
)
flags.DEFINE_float(
    'learning_rate',
    0.0001,
    'The learning rate to use in the initial phase of training',
)
flags.DEFINE_enum(
    'optimizer',
    'RMSprop',
    ['RMSprop', 'adam'],
    'Optimizater to use for traning',
)
flags.DEFINE_integer(
    'image_size',
    224,
    'Height and width of the images input into the network',
)
flags.DEFINE_string(
    'train_log_dir',
    'logs/modGen',
    'Base directory to log training and validation info to',
)
flags.DEFINE_string(
    'model_dir',
    'savedModels',
    'Base directory to save trined models',
)
flags.DEFINE_bool(
    'use_multi_processing',
    True,
    'Use multi-processing for training',
)
flags.DEFINE_bool(
    'send_to_cloud',
    True,
    'Send the saved model to S3 storage',
)

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

def bucket_exists(bucket_name):
    """Determine whether bucket_name exists and the user has permission to access it
    :param bucket_name: string
    :return: True if the referenced bucket_name exists, otherwise False
    """
    s3 = boto3.client('s3')
    try:
        response = s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_files(bucket, files_to_send):
    s3 = boto3.client('s3')
    for file_index, file_to_send in enumerate(files_to_send):
        print("Uploading file to %s:%s, %i/%i" % \
            (bucket, file_to_send, file_index + 1, len(files_to_send)))
        try:
            s3.upload_file(file_to_send, bucket, file_to_send,\
                Callback=ProgressPercentage(file_to_send))
        except ClientError as e:
            print(e)
        print()

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


def run_training():
    start_time = int(time.time())
   
    # determine if gpu is being used
    gpu_avail = tf.test.is_gpu_available(cuda_only=True)
    cuda_build = tf.test.is_built_with_cuda()

    print('GPU available: %r, Built with CUDA: %r' % (gpu_avail, cuda_build))

    saved_model_dir = flags.FLAGS.model_dir + '/' + str(start_time)

    initial_log_dir = flags.FLAGS.train_log_dir + '/' + str(start_time) + '/initial'

    IMG_SHAPE = (flags.FLAGS.image_size, flags.FLAGS.image_size, 3)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        horizontal_flip=True,
        validation_split=flags.FLAGS.val_split)

    train_generator = datagen.flow_from_directory(
        flags.FLAGS.input_dir,
        target_size=(flags.FLAGS.image_size, flags.FLAGS.image_size),
        batch_size=flags.FLAGS.batch_size,
        shuffle=True,
        subset='training')

    val_generator = datagen.flow_from_directory(
        flags.FLAGS.input_dir,
        target_size=(flags.FLAGS.image_size, flags.FLAGS.image_size),
        batch_size=flags.FLAGS.batch_size,
        shuffle=True,
        subset='validation')

    labels = '\n'.join(sorted(train_generator.class_indices.keys()))
    
    print('Labels:')
    print(train_generator.class_indices)
    create_output_dir(flags.FLAGS.model_dir)    
    create_output_dir(saved_model_dir)
    
    with open(saved_model_dir + '/labels.txt', 'w') as f:
        f.write(labels)
    
    callback_init = tf.keras.callbacks.TensorBoard(
        initial_log_dir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.Dropout(flags.FLAGS.dropout),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    opt = None
    if(flags.FLAGS.optimizer=='adam'):
        opt = tf.keras.optimizers.Adam(flags.FLAGS.learning_rate)
    else:
        opt = tf.keras.optimizers.RMSprop(flags.FLAGS.learning_rate)
    
    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'],)

    initial_history = model.fit_generator(
        train_generator,
        epochs=flags.FLAGS.num_epochs,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples//val_generator.batch_size,
        use_multiprocessing=flags.FLAGS.use_multi_processing,
        workers= 8 if flags.FLAGS.use_multi_processing else 1,
        shuffle=True,
        callbacks=[callback_init])

    if not flags.FLAGS.fine_tune:
        model.save(saved_model_dir)
        return saved_model_dir

    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    fine_tune_log_dir = flags.FLAGS.train_log_dir + '/' + str(start_time) + '/fine-tune'

    callback_ft = tf.keras.callbacks.TensorBoard(
        fine_tune_log_dir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    if(flags.FLAGS.optimizer=='adam'):
        opt = tf.keras.optimizers.Adam(flags.FLAGS.learning_rate/10)
    else:
        opt = tf.keras.optimizers.RMSprop(flags.FLAGS.learning_rate/10)
 
    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'],)

    ft_history = model.fit_generator(
        train_generator,
        epochs=flags.FLAGS.num_epochs * 2,
        initial_epoch=flags.FLAGS.num_epochs,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples//val_generator.batch_size,
        callbacks=[callback_ft])

    model.save(saved_model_dir)
    return saved_model_dir

def main(unused_argv):
    
    if flags.FLAGS.send_to_cloud:
        # Check if the s3 bucket exists
        if bucket_exists(s3_bucket_name):
            print('%s exists and you have permission to access it.' % s3_bucket_name)
        else:
            print('%s does not exist or you do not have permission to access it.' \
                    % s3_bucket_name)
            sys.exit(0)

    saved_model_dir = run_training()
    if flags.FLAGS.send_to_cloud:
        # Zip contents of saved model dir and send to s3
        zip_file_name = shutil.make_archive(saved_model_dir, 'zip', saved_model_dir)
        upload_files(s3_bucket_name, ['%s.zip' % saved_model_dir])
        os.remove(zip_file_name)        

if __name__ == "__main__":
  app.run(main)
