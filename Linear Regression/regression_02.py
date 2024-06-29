import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

plt.scatter(X, y)
plt.show()

print(f"Value of X: {X}")
print(f"Value of y: {y}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")

X_train = X[:40]  # First 40 examples for training set
y_train = y[:40]

X_test = X[40:]  # Last 10 as test examples
y_test = y[40:]

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', label='Training Data')
plt.scatter(X_test, y_test, c='g', label='Testing Data')
plt.legend()
plt.show()

# Set random seed
tf.random.set_seed(42)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mae',
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

print(model.summary())

history = model.fit(X_train, y_train, epochs=100)

plt.plot(history.history['loss'])
plt.show()

y_preds = model.predict(X_test)
print(f"Predicted values: {y_preds}")
print(f"Actual values: {y_test}")


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', label='Training Data')
    plt.scatter(test_data, test_labels, c='g', label='Testing Data')
    plt.scatter(test_data, predictions, c='r', label='Predictions')
    plt.legend()
    plt.show()


plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Second model 2 layers trained for 100 epochs

tf.random.set_seed(42)

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss='mae',
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

model_2.fit(X_train, y_train, epochs=100)
y_preds_2 = model_2.predict(X_test)
print(f"Predicted values: {y_preds_2}")
print(f"Actual values: {y_test}")
plot_predictions(X_train, y_train, X_test, y_test, y_preds_2)
