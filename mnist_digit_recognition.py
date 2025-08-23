import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels (0–9 → [0,0,1,0...])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build model
model = Sequential([
    Flatten(input_shape=(28,28)),      # Flatten 28x28 → 784
    Dense(128, activation='relu'),     # Hidden layer
    Dense(64, activation='relu'),      # Hidden layer
    Dense(10, activation='softmax')    # Output layer (10 classes)
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Predict a sample digit
import numpy as np
sample = X_test[0].reshape(1,28,28)
prediction = model.predict(sample)
print("Predicted Digit:", np.argmax(prediction))
