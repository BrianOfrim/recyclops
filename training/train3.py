import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

IMAGE_SIZE = 224
BATCH_SIZE = 64

base_dir = './images'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')

for image_batch, label_batch in train_generator:
    break
print("Image batch")
print(image_batch.shape)
print(label_batch.shape)
print (train_generator.class_indices)

print(label_batch[0])

class_names = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

for i in range(2):
    label_id = np.argmax(label_batch[i], axis=-1)
    print(class_names[label_id])

    plt.figure()
    plt.imshow(image_batch[i])
    plt.title(class_names[label_id])


plt.show()
