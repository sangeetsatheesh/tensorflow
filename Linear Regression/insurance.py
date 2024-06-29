"""
Predict the cost of medical insurance of an individual based on various parameters
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

insurance_data = pd.read_csv(
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

print(insurance_data.head())
print(insurance_data.loc[0])
# Turn all categories into numbers
# insurance_one_hot = pd.get_dummies(insurance_data)
# print(insurance_one_hot.head())
# X_one_hot = insurance_one_hot.drop("charges", axis=1)
# y_one_hot = insurance_one_hot["charges"]

ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)
X = insurance_data.drop("charges", axis=1)
y = insurance_data["charges"]

print(f"Training data head:\n", X.head())
print(f"Testing data head:\n", y.head())

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
# Fit the column transformer only on the training data
ct.fit(X_train)

# Transform training and test data with normalization and one hot encoding
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(f"X train loc 0", X_train.loc[0])
print(f"X train normalized and one hot encoded example\n{X_train_normal[0]}")
print(f"X train shape is {X_train.shape}")
print(f"X train normal shape is {X_train_normal.shape}")

# Set random seed
tf.random.set_seed(42)

# Create a new model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['mae'])
print(insurance_model.summary())

history = insurance_model.fit(X_train_normal, y_train, epochs=200)
plt.plot(history.history['loss'])
plt.show()

print(insurance_model.evaluate(X_test_normal, y_test))
