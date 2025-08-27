import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load datasetdf = pd.read_csv(r"D:\shelly\ml_practice\Titanic_Survival_Prediction\titanic.csv")
df = pd.read_csv(r"D:\shelly\ml_practice\Titanic_Survival_Prediction\titanic.csv")


# 2. Preprocessing
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Age"] = df["Age"].fillna(df["Age"].median())
X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. Evaluate
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save the model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("✅ Model saved as titanic_model.pkl")
