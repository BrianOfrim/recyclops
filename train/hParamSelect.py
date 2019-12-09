import os
from sys import exit
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import itertools as it

IMAGE_SIZE = 224
VAL_SPLIT = 0.2
NUM_INITIAL_EPOCHS = 25
NUM_FINE_TUNE_EPOCHS = 10


HPSet = collections.namedtuple('HPSet', ['batch_size','dropout','optimizer', 'base_learning_rate','fine_tune'])

hpvals = [
    HPSet(4, 0.45, 'adam', 0.0001, False),
    HPSet(8, 0, 'adam', 0.0001, False),
    HPSet(8, 0.45, 'adam', 0.00001, False),
    HPSet(8, 0.25, 'adam', 0.0001, False),
    HPSet(32, 0, 'adam', 0.0001, False),
    HPSet(8, 0.45, 'RMSprop', 0.00001, False),
    HPSet(4, 0, 'adam', 0.00001, False),
    HPSet(16, 0.25, 'adam', 0.00001, False),
    HPSet(8, 0.25, 'RMSprop', 0.00001, False),
    HPSet(16, 0, 'adam', 0.0001, False),
    HPSet(4, 0.25, 'RMSprop', 0.00001, False),
    HPSet(32, 0.45, 'adam', 0.0001, False)
]

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([4]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_BASE_LEARNING_RATE = hp.HParam('base_learning_rate', hp.Discrete(['0.0001']))
HP_FINE_TUNE = hp.HParam('do_fine_tune', hp.Discrete([False]))


HPARAMS = [
    HP_BATCH_SIZE,
    HP_DROPOUT,
    HP_OPTIMIZER,
    HP_BASE_LEARNING_RATE,
    HP_FINE_TUNE,
]

METRIC_ACCURACY = 'accuracy'

METRICS = [
    hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="accuracy (val)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val)",
    ),
    hp.Metric(
        "epoch_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "epoch_loss",
        group="train",
        display_name="loss (train)",
    ),
]

base_dir = './images'
log_dir = 'logs/hparam_tuning/%i/' % int(time.time())

with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=HPARAMS,
    metrics=METRICS,
  )

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    horizontal_flip=True,
    validation_split=VAL_SPLIT)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

def train_test_model(run_dir, hparams):

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    train_generator = datagen.flow_from_directory(
        directory=base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=hparams[HP_BATCH_SIZE], 
        shuffle=True,
        subset='training')

    val_generator = datagen.flow_from_directory(
        directory=base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=hparams[HP_BATCH_SIZE], 
        subset='validation')
    
    initial_dir = run_dir + '-initial'
    
    callback_init = tf.keras.callbacks.TensorBoard(
        initial_dir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback_init = hp.KerasCallback(initial_dir, hparams)

    model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    opt = None
    if(hparams[HP_OPTIMIZER]=='adam'):
        opt = tf.keras.optimizers.Adam(hparams[HP_BASE_LEARNING_RATE])
    if(hparams[HP_OPTIMIZER]=='RMSprop'):
        opt = tf.keras.optimizers.RMSprop(hparams[HP_BASE_LEARNING_RATE])

    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=[METRIC_ACCURACY])

    initial_history = model.fit_generator(
        train_generator,
        epochs=NUM_INITIAL_EPOCHS,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples//val_generator.batch_size,
        callbacks=[callback_init, hparams_callback_init])


    return # do not run fine tuning

    hparams[HP_FINE_TUNE] = True

    if(hparams[HP_OPTIMIZER]=='adam'):
        opt = tf.keras.optimizers.Adam(hparams[HP_BASE_LEARNING_RATE]/10)
    if(hparams[HP_OPTIMIZER]=='RMSprop'):
        opt = tf.keras.optimizers.RMSprop(hparams[HP_BASE_LEARNING_RATE]/10)

    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    fine_tune_dir = run_dir + '-fine_tune'

    callback_ft = tf.keras.callbacks.TensorBoard(
        fine_tune_dir,
        update_freq='epoch',
        profile_batch=0,  # workaround for issue #2084
    )

    hparams_callback_ft = hp.KerasCallback(fine_tune_dir, hparams)

    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=[METRIC_ACCURACY])

    ft_history = model.fit_generator(
        train_generator,
        epochs=NUM_INITIAL_EPOCHS+NUM_FINE_TUNE_EPOCHS,
        initial_epoch=NUM_INITIAL_EPOCHS,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples//val_generator.batch_size,
        callbacks=[callback_ft, hparams_callback_ft])
 

def run(run_dir, hparams):
    train_test_model(run_dir, hparams)

session_num = 0
for hpval in hpvals:
    hparams = {
        HP_BATCH_SIZE: hpval.batch_size,
        HP_DROPOUT: hpval.dropout,
        HP_OPTIMIZER: hpval.optimizer,
        HP_BASE_LEARNING_RATE: hpval.base_learning_rate,
        HP_FINE_TUNE: hpval.fine_tune,
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run( log_dir + run_name, hparams)
    session_num += 1
