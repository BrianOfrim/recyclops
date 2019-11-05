import os
from sys import exit
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_EPOCHS = 10

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 32, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','RMSprop']))

METRIC_ACCURACY = 'accuracy'

base_dir = './images'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_BATCH_SIZE, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    #width_shift_range=.15,
    #height_shift_range=.15,
    horizontal_flip=True,
    #zoom_range=0.5,
    validation_split=VAL_SPLIT)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

def train_test_model(hparams):

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
    
    model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    model.fit_generator(
        train_generator,
        epochs= NUM_EPOCHS)
    
    print('Evaluating...')
    _, accuracy = model.evaluate_generator(val_generator, verbose=1)

    return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

start_time = int(time.time())

for batch_size in HP_BATCH_SIZE.domain.values:
    for dropout in HP_DROPOUT.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_BATCH_SIZE: batch_size,
                HP_DROPOUT: dropout,
                HP_OPTIMIZER: optimizer
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + str(start_time) + '/' + run_name, hparams)
            session_num += 1
