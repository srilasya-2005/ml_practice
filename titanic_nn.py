# titanic_nn.py
# Neural Network (Keras) demo on Titanic dataset (mini version)
# Part 7 of ML crash course

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# --- Small Titanic dataset (manual) ---
data = {
    "Pclass": [3,1,3,1,3,3,1,3,2,3,1,3,3,3,1,3,1,2,3,3],
    "Sex":    ["male","female","female","female","male","male","male","male","female","female",
               "female","male","female","male","male","male","female","male","male","female"],
    "Age":    [22,38,26,35,35,54,2,27,14,4,58,20,39,14,55,2,30,31,40,15],
    "Survived":[0,1,1,1,0,0,0,1,1,1,0,1,0,0,0,0,1,0,1,1]
}

df = pd.DataFrame(data)

# Preprocessing
X = df.drop("Survived", axis=1).copy()
y = df["Survived"]

# Encode categorical
X["Sex"] = X["Sex"].map({"male":0,"female":1})
X["Age"] = X["Age"].fillna(X["Age"].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Build Neural Network ---


model = Sequential([
    Input(shape=(X.shape[1],)),   # safer way
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
print("\nTraining Neural Network...")
history = model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nNeural Network Accuracy on test set: {acc:.2f}")

# Predict on a new passenger
sample = np.array([[3, 0, 25]])  # Pclass=3, male, Age=25
pred = model.predict(sample)[0][0]
print("\nPrediction for sample passenger (Pclass=3, male, Age=25):")
print("Survived" if pred > 0.5 else "Not Survived", f"(prob={pred:.2f})")
