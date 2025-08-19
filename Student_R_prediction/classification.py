import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = pd.read_csv("student_pass.csv")
print("Dataset Preview:\n", data.head())  # 👈 check columns

# Features and target
X = data[['Hours']]   # independent variable
y = data['Pass']      # dependent variable

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Manual test
hours = 4
result = model.predict(pd.DataFrame([[hours]], columns=['Hours']))[0]

print(f"\nIf a student studies {hours} hours → {'Pass' if result==1 else 'Fail'}")
