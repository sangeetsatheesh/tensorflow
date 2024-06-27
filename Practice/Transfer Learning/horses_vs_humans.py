import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import concatenate

train_dir = 'data/training'
validation_dir = 'data/validation'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')
print(f"There are {len(os.listdir(train_horses_dir))} images of horses for training")
print(f"There are {len(os.listdir(train_humans_dir))} images of humans for training")
print(f"There are {len(os.listdir(validation_horses_dir))} images of horses for validation")
print(f"There are {len(os.listdir(validation_humans_dir))} images of humans for validation")

print("Sample horse image:")
plt.imshow(load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}"))
plt.title("Horse")
plt.axis(False)
plt.show()

print("\nSample human image:")
plt.imshow(load_img(f"{os.path.join(train_humans_dir, os.listdir(train_humans_dir)[0])}"))
plt.title("Human")
plt.axis(False)
plt.show()

sample_image = load_img(f"{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}")

# Convert the image into its numpy array representation
sample_array = img_to_array(sample_image)

print(f"Each image has shape: {sample_array.shape}")


def train_val_generators(training_dir, validation_dir):
    """
    Creates the training and validation data generators
    :param training_dir:
    :param validation_dir:
    :return:
    """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.,
                                       rotation_range=0.1,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=0.1,
                                       fill_mode='reflect')
    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                  batch_size=32,
                                                                  class_mode='binary',
                                                                  target_size=(150, 150))
    return train_generator, validation_generator


train_generator, validation_generator = train_val_generators(train_dir, validation_dir)
"""
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
"""
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_pre_trained_model(local_weights_file):
    """
    Initializes an InceptionV3 model
    :param local_weights_file:
    :return:
    """
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=local_weights_file)
    pre_trained_model.load_weights(local_weights_file)
    # Make all the layers in the pretrained model non trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False
    return pre_trained_model


pre_trained_model = create_pre_trained_model(local_weights_file)
pre_trained_model.summary()

total_params = pre_trained_model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


def output_of_last_layer(pre_trained_model):
    """
    Gets the last layer output of a model
    :param pre_trained_model:
    :return:
    """
    last_desired_layer = pre_trained_model.get_layer("mixed7")
    print('Last layer output shape: ', last_desired_layer)
    lastl_output = last_desired_layer.output
    print('last layer output: ', lastl_output)
    return lastl_output


last_output = output_of_last_layer(pre_trained_model)

print(f"The pretrained model has type: {type(pre_trained_model)}")


def create_final_model(pre_trained_model, last_output):
    """
    Appends a custom model to a pre-trained_model
    :param pre_trained_model:
    :param last_output:
    :return:
    """
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1024 hidden units and ReLU activation
    x = layers.Dense(1024, activation="relu")(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)
    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation="sigmoid")(x)
    # Create the complete model by using the Model class
    model = Model(inputs=pre_trained_model.input, outputs=x)
    # Compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


model = create_final_model(pre_trained_model, last_output)

# Inspect parameters
total_params = model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")

callbacks = myCallback()
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    verbose=2,
                    callbacks=callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
