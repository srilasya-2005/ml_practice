import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_scores.csv")
print("Dataset Preview:")
print(data.head())

# Split data
X = data[['Hours']]  # feature
y = data['Score']    # target

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
hours = 6
pred = model.predict([[hours]])
print(f"\nIf a student studies {hours} hours, predicted score = {pred[0]:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")
plt.legend()
plt.show()
