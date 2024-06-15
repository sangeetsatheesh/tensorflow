"""
Use food 101 dataset cleaned by mrdbourke and use that to classify
if a picture contains a pizza or a steak
"""

import tensorflow as tf
import zipfile
import subprocess
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# from keras.preprocessing.image import ImageDataGenerator


# subprocess.call(["wget", "https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"])
#
# zip_ref = zipfile.ZipFile("pizza_steak.zip","r")
# zip_ref.extractall()
# zip_ref.close()

# for dirpath, dirnames, filenames in os.walk("pizza_steak"):
#     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
#
# num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
#
# print(f"Number of images in training set: ", num_steak_images_train)
#
# data_dir = pathlib.Path("pizza_steak/train/")  # turn our training path into a Python path
# class_names = np.array(
#     sorted([item.name for item in data_dir.glob('*')]))  # created a list of class_names from the subdirectories
# print(f"Class names found are:\n", class_names)
#

# def view_random_image(target_dir, target_class):
#     target_folder = target_dir + target_class
#     print(os.listdir(target_folder))
#     random_image = random.sample(os.listdir(target_folder), 1)
#     img = mpimg.imread(target_folder + "/" + random_image[0])
#     plt.imshow(img)
#     plt.title(target_class)
#     plt.axis("off")
#     plt.show()
#     print(f"Image shape: {img.shape}")
#     return img
#
#
# rand_img = view_random_image("pizza_steak/train/", class_names[1])

tf.random.set_seed(42)

# Preprocess data
#
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# val_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

# train_data = train_datagen.flow_from_directory(train_dir,
#                                                target_size=(224, 224),
#                                                batch_size=32,
#                                                class_mode='binary',
#                                                seed=42)
# valid_data = val_datagen.flow_from_directory(test_dir,
#                                              target_size=(224, 224),
#                                              batch_size=32,
#                                              class_mode='binary',
#                                              seed=42)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=42,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
print(f"First Image Shape: {image_batch[0].shape}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


print(f"Train data shape: {train_ds}")
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.summary()

history_1 = model_1.fit(train_ds,
                        epochs=5,
                        validation_data=test_ds)



model_2 = tf.keras.models.Sequential([

])