import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ===================
# 1. Train model
# ===================
df = pd.read_csv(r"D:\shelly\ml_practice\Titanic_Survival_Prediction\titanic.csv")


# Keep only useful columns
X = df[["Pclass", "Sex", "Age"]].copy()
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X["Age"] = X["Age"].fillna(X["Age"].median())
y = df["Survived"]

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("titanic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ===================
# 2. Flask API
# ===================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract passenger details
    pclass = data.get("Pclass")
    sex = 0 if data.get("Sex") == "male" else 1
    age = data.get("Age")

    # Make prediction
    features = [[pclass, sex, age]]
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": round(float(prob), 2),
        "message": "Survived" if prediction == 1 else "Not Survived"
    })

if __name__ == "__main__":
    app.run(debug=True)
