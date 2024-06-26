import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

source_path = 'data/PetImages'
source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")

root_dir = 'data/cats-v-dogs'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
    shutil.rmtree(root_dir)


def create_train_val_dirs(root_path):
    """
    Creates directories for the train and test sets

    Args:
      root_path (string) - the base directory path to create subdirectories from

    Returns:
      None
    """
    os.makedirs(os.path.join(root_path, "training/cats"))
    os.makedirs(os.path.join(root_path, "training/dogs"))
    os.makedirs(os.path.join(root_path, "validation/cats"))
    os.makedirs(os.path.join(root_path, "validation/dogs"))


try:
    create_train_val_dirs(root_path=root_dir)
except FileExistsError:
    print("You should not be seeing this since the upper directory is removed beforehand")
for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    """
    Splits the data into train and test sets

    Args:
      SOURCE_DIR (string): directory path containing the images
      TRAINING_DIR (string): directory path to be used for training
      VALIDATION_DIR (string): directory path to be used for validation
      SPLIT_SIZE (float): proportion of the dataset to be used for training

    Returns:
      None
    """

    file_list = os.listdir(SOURCE_DIR)
    for data_file in file_list:
        if os.path.getsize(os.path.join(SOURCE_DIR, data_file)) <= 0:
            print(data_file + " is zero length, so ignoring.")
            file_list.remove(data_file)
    file_length = len(file_list)
    split = int(file_length * SPLIT_SIZE)

    file_list = random.sample(file_list, file_length)

    for train_file in file_list[0:int(file_length * SPLIT_SIZE)]:
        copyfile(os.path.join(SOURCE_DIR, train_file), os.path.join(TRAINING_DIR, train_file))

    for val_file in file_list[int(file_length * SPLIT_SIZE):]:
        copyfile(os.path.join(SOURCE_DIR, val_file), os.path.join(VALIDATION_DIR, val_file))


# Define paths
CAT_SOURCE_DIR = "data/PetImages/Cat/"
DOG_SOURCE_DIR = "data/PetImages/Dog/"

TRAINING_DIR = "data/cats-v-dogs/training/"
VALIDATION_DIR = "data/cats-v-dogs/validation/"

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

# Empty directories in case you run this cell multiple times
if len(os.listdir(TRAINING_CATS_DIR)) > 0:
    for file in os.scandir(TRAINING_CATS_DIR):
        os.remove(file.path)
if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
    for file in os.scandir(TRAINING_DOGS_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
    for file in os.scandir(VALIDATION_CATS_DIR):
        os.remove(file.path)
if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
    for file in os.scandir(VALIDATION_DOGS_DIR):
        os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# Run the function
# NOTE: Messages about zero length images should be printed out
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Your function should perform copies rather than moving images so original directories should contain unchanged images
print(f"\n\nOriginal cat's directory has {len(os.listdir(CAT_SOURCE_DIR))} images")
print(f"Original dog's directory has {len(os.listdir(DOG_SOURCE_DIR))} images\n")

# Training and validation splits
print(f"There are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")


def train_val_generators(training_dir, validation_dir):
    """
    Creates the training and validation data generators

    Args:
      training_dir (string): directory path containing the training images
      validation_dir (string): directory path containing the testing/validation images

    Returns:
      train_generator, validation_generator - tuple containing the generators
    """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.,
                                       rotation_range=0.1,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=0.1,
                                       fill_mode=0.1)

    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.,
                                            rotation_range=0.1,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            shear_range=0.1,
                                            zoom_range=0.1,
                                            horizontal_flip=0.1,
                                            fill_mode=0.1)

    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                  batch_size=32,
                                                                  class_mode='binary',
                                                                  target_size=(150, 150))
    return train_generator, validation_generator


train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)
