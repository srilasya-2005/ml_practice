from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("titanic_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")   # render the HTML form

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form  # coming from HTML form
    pclass = int(data["Pclass"])
    sex = 1 if data["Sex"] == "female" else 0
    age = float(data["Age"])

    X_new = pd.DataFrame([[pclass, sex, age]], columns=["Pclass", "Sex", "Age"])
    prediction = model.predict(X_new)[0]

    result = "Survived" if prediction == 1 else "Not Survived"
    return render_template("index.html", prediction_text=f"Passenger would: {result}")

if __name__ == "__main__":
    app.run(debug=True)
