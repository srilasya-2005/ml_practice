import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset (study hours, pass/fail)
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

X = df[['Hours']].values
y = df['Pass'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural Network
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))  # hidden layer
model.add(Dense(1, activation='sigmoid'))            # output layer (0 or 1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy on test data: {acc:.2f}")

# Prediction
hours = 4
pred = model.predict(np.array([[hours]]))
print(f"If a student studies {hours} hours → {'Pass' if pred[0][0] > 0.5 else 'Fail'}")
