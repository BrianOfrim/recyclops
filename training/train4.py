import os
from sys import exit
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

IMAGE_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2

base_dir = './images'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=VAL_SPLIT)

train_generator = datagen.flow_from_directory(
    directory=base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    shuffle=True,
    subset='training')

val_generator = datagen.flow_from_directory(
    directory=base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

print('Traing set size: %i' % train_generator.samples)
print('Validation set size: %i' % val_generator.samples)

print('Training set batch size: %i' % train_generator.batch_size)
print('Validation set batch size: %i' % val_generator.batch_size)

print('Total dataset size: %i' % (train_generator.samples + val_generator.samples))

for image_batch, label_batch in train_generator:
  break
print('Image batch shape:')
print(image_batch.shape)
print('Label batch shape:')
print(label_batch.shape)

print (train_generator.class_indices)
print(len(train_generator.class_indices))
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

feature_batch = base_model(image_batch)

print(feature_batch.shape)

base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
#  tf.keras.layers.Conv2D(64, 3, activation='relu'),  
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(len(train_generator.class_indices), activation='softmax')

])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

print(len(model.trainable_variables))

training_steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps_per_epoch = val_generator.samples // val_generator.batch_size

print('Training steps per epoch: %i' % training_steps_per_epoch)
print('Validaiton steps per epoch: %i' % validation_steps_per_epoch)
initial_epochs = 15

history = model.fit_generator(train_generator,
                    epochs=initial_epochs,
                    steps_per_epoch=training_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=validation_steps_per_epoch)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
#plt.show()


base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()

print('Trainable var: %i' % len(model.trainable_variables))

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit_generator(train_generator,
                         epochs=total_epochs,
                         initial_epoch = initial_epochs,
                         validation_data=val_generator)


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

