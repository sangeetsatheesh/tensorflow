import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

""" Dropped tf.keras.datasets because of error 
ImportError: cannot import name '_initialize_variables' from 'keras.src.backend' """
from keras.datasets import fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

print(f"Training Samples length: {len(train_data)}")
print(f"Training Labels length: {len(train_labels)}")
print(f"Training Samples 0th :\n{train_data[0]}")
print(f"Training Labels 0th : {train_labels[0]}")

print(f"Train data shape: {train_data.shape}")
print(f"Train Labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Test Labels shape: {test_labels.shape}")

# Plot a single example
plt.imshow(train_data[0])
plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot an example image and its label
plt.imshow(train_data[17], cmap=plt.cm.binary)  # change the colours to black & white
plt.title(class_names[train_labels[17]])
plt.show()

# Set random seed
tf.random.set_seed(42)

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_1.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

non_norm_history = model_1.fit(train_data,
                               train_labels,
                               epochs=10,
                               validation_data=(test_data, test_labels))

pd.DataFrame(non_norm_history.history).plot(title="With Non Normalized Data")
plt.show()

train_data = train_data / 255.0
test_data = test_data / 255.0

tf.random.set_seed(42)

model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_2.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

norm_history = model_2.fit(train_data,
                           train_labels,
                           epochs=10,
                           validation_data=(test_data, test_labels))
pd.DataFrame(norm_history.history).plot(title="With Normalized Data")
plt.show()
