import os
from sys import exit
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import itertools as it

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_INITIAL_EPOCHS = 10
NUM_FINE_TUNE_EPOCHS = 10

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','RMSprop']))
HP_BASE_LEARNING_RATE = hp.HParam('base_learning_rate', hp.Discrete([0.001, 0.0001]))
HP_FINE_TUNE = hp.HParam('do_fine_tune', hp.Discrete([True, False]))

METRIC_ACCURACY = 'accuracy'

HPARAMS = [
    HP_BATCH_SIZE,
    HP_DROPOUT,
    HP_OPTIMIZER,
    HP_BASE_LEARNING_RATE,
    HP_FINE_TUNE,
]

METRICS = [
    hp.Metric(
        METRIC_ACCURACY,
        display_name='Accuracy'
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

    opt = None
    if(hparams[HP_OPTIMIZER]=='adam'):
        opt = tf.keras.optimizers.Adam(hparams[HP_BASE_LEARNING_RATE])
    if(hparams[HP_OPTIMIZER]=='RMSprop'):
        opt = tf.keras.optimizers.RMSprop(hparams[HP_BASE_LEARNING_RATE])

    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=NUM_INITIAL_EPOCHS)
    
    if(not hparams[HP_FINE_TUNE]):
        print('Evaluating...')
        _, accuracy = model.evaluate_generator(val_generator, verbose=1)
        return accuracy
    
    if(hparams[HP_OPTIMIZER]=='adam'):
        opt = tf.keras.optimizers.Adam(hparams[HP_BASE_LEARNING_RATE]/10)
    if(hparams[HP_OPTIMIZER]=='RMSprop'):
        opt = tf.keras.optimizers.RMSprop(hparams[HP_BASE_LEARNING_RATE]/10)

    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(
      optimizer=opt,
      loss='categorical_crossentropy',
      metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        epochs=NUM_INITIAL_EPOCHS+NUM_FINE_TUNE_EPOCHS,
        initial_epoch=NUM_INITIAL_EPOCHS)

    print('Evaluating...')
    _, accuracy = model.evaluate_generator(val_generator, verbose=1)
    return accuracy
 

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for batch_size in HP_BATCH_SIZE.domain.values:
    for dropout in HP_DROPOUT.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for base_learning_rate in HP_BASE_LEARNING_RATE.domain.values:
                for fine_tune in HP_FINE_TUNE.domain.values:
                    hparams = {
                        HP_BATCH_SIZE: batch_size,
                        HP_DROPOUT: dropout,
                        HP_OPTIMIZER: optimizer,
                        HP_BASE_LEARNING_RATE: base_learning_rate,
                        HP_FINE_TUNE: fine_tune,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run( log_dir + run_name, hparams)
                    session_num += 1
