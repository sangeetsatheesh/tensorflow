"""
Sign Language MNIST
"""
import csv
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import gdown

# Sign MNIST train csv
gdown.download(id='1z0DkA9BytlLxO1C0BAWzknLyQmZAp0HR')
# Sign MNIST test csv
gdown.download(id='1z1BIj4qmri59GWBG4ivMNFtpZ4AXIbzg')

TRAINING_FILE = './sign_mnist_train.csv'
VALIDATION_FILE = './sign_mnist_test.csv'

with open(TRAINING_FILE) as training_file:
    line = training_file.readline()
    print(f"First line (header) looks like this:\n{line}")
    line = training_file.readline()
    print(f"Each subsequent line (data points) look like this:\n{line}")


def parse_data_from_input(filename):
    """
    Parses the images and labels from a csv file
    :param filename:
    :return:
    """
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)

        labels = []
        images = []
        for row in csv_reader:
            labels.append(row[0])
            single_image = row[1:785]
            single_image = np.array_split(single_image, 28)
            images.append(single_image)

        images = np.array(images).astype(float)
        labels = np.array(labels).astype(float)

    return images, labels


training_images, training_labels = parse_data_from_input(TRAINING_FILE)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")


def plot_categories(training_images, training_labels):
    fig, axes = plt.subplots(1, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    for k in range(10):
        img = training_images[k]
        img = np.expand_dims(img, axis=-1)
        img = array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap="Greys_r")
        ax.set_title(f"{letters[int(training_labels[k])]}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


plot_categories(training_images, training_labels)


def train_val_generators(training_images,
                         training_labels,
                         validation_images,
                         validation_labels):
    """
    Creates the training and validation data generators
    :param training_images:
    :param training_labels:
    :param validation_images:
    :param validation_labels:
    :return:
    """
    training_images = np.expand_dims(training_images, axis=3)
    validation_images = np.expand_dims(validation_images, axis=3)
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=30,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       shear_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=32)
    validation_datagen = ImageDataGenerator(rescale=1. / 255.)
    validation_generator = validation_datagen.flow(x=validation_images,
                                                   y=validation_labels,
                                                   batch_size=32)
    return train_generator, validation_generator


train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images,
                                                             validation_labels)

print(f"Images of training generator have shape: {train_generator.x.shape}")
print(f"Labels of training generator have shape: {train_generator.y.shape}")
print(f"Images of validation generator have shape: {validation_generator.x.shape}")
print(f"Labels of validation generator have shape: {validation_generator.y.shape}")


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.Dense(units=26, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = create_model()

# Train your model
history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

