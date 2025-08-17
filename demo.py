# Basic ML with scikit-learn
from sklearn.linear_model import LinearRegression
import numpy as np

# Data (area in sq.ft, price in $1000s)
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200, 250, 300, 350, 400])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict price for 2200 sq.ft
prediction = model.predict([[2200]])
print("Predicted price:", prediction[0])
