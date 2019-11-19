import os.path
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

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
    './logs/modGen',
    'Base directory to log training and validation info to',
)
flags.DEFINE_string(
    'model_dir',
    './savedModels',
    'Base directory to save trined models',
)

def run_training():
    start_time = int(time.time())
    
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
        subset='training')

    val_generator = datagen.flow_from_directory(
        flags.FLAGS.input_dir,
        target_size=(flags.FLAGS.image_size, flags.FLAGS.image_size),
        batch_size=flags.FLAGS.batch_size, 
        subset='validation')

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
        callbacks=[callback_init])

    if not flags.FLAGS.fine_tune:
        tf.saved_model.save(model, flags.FLAGS.model_dir + '/' + str(start_time))
        return

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

    tf.saved_model.save(model, flags.FLAGS.model_dir + '/' + str(start_time))

def main(unused_argv):
    run_training()

if __name__ == "__main__":
  app.run(main)
