import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("titanic.csv")

# Select features
features = ["Pclass", "Sex", "Age", "Fare"]
X = data[features]
y = data["Survived"]

# Preprocess (convert 'Sex' to numbers, fill missing Age)
X = X.copy()  # make a safe copy
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Age"] = X["Age"].fillna(X["Age"].median())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Try manual prediction
sample = pd.DataFrame([[3, 0, 25, 7.25]], columns=features)  
# 3rd class, male, age 25, fare 7.25
print("Prediction for sample passenger:",
      "Survived" if rf.predict(sample)[0]==1 else "Not Survived")
