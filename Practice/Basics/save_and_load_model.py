import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

plt.scatter(X, y)
plt.show()

print(f"Value of X: {X}")
print(f"Value of y: {y}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Set random seed as 42
tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

print(f"Model Summary\n", model.summary())
# Fit the model
history = model.fit(X, y, epochs=100)

plt.plot(history.history['loss'])
plt.show()

model.export('model_savedmodel_format')
model.save('model.h5')
loaded_model = tf.keras.models.load_model('model.h5')
print(f"Loaded model summary\n", loaded_model.summary())
