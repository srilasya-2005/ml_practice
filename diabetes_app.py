from flask import Flask, request, render_template_string
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Binarizer
import numpy as np

# -------------------
# Prepare dataset
# -------------------
# sklearn diabetes dataset is regression (target is continuous)
# We'll convert it into classification: >140 means "diabetes", else "no diabetes"
diabetes = load_diabetes()
X = diabetes.data
y = (diabetes.target > 140).astype(int)  # 1 = diabetes, 0 = no diabetes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------
# Flask App
# -------------------
app = Flask(__name__)

# Inline HTML form
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction</title>
</head>
<body>
    <h2>Diabetes Prediction Form</h2>
    <form method="POST" action="/predict">
        <label>BMI (Body Mass Index):</label>
        <input type="number" step="0.1" name="bmi" required><br><br>

        <label>Blood Pressure:</label>
        <input type="number" step="0.1" name="bp" required><br><br>

        <label>Age:</label>
        <input type="number" step="1" name="age" required><br><br>

        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h2>Prediction Result</h2>
    <p>The model predicts: <b>{{ result }}</b></p>
    <a href="/">Go Back</a>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(form_html)

@app.route("/predict", methods=["POST"])
def predict():
    bmi = float(request.form['bmi'])
    bp = float(request.form['bp'])
    age = float(request.form['age'])

    # Map inputs to 3 features from diabetes dataset: BMI, BP, Age
    # (these correspond to indices 2, 3, 0 roughly scaled)
    # We'll just fill a vector with zeros except our inputs
    features = np.zeros((1, X.shape[1]))
    features[0, 2] = bmi / 50   # scale approx
    features[0, 3] = bp / 200
    features[0, 0] = age / 100

    prediction = model.predict(features)[0]
    result = "Diabetes" if prediction == 1 else "No Diabetes"

    return render_template_string(result_html, result=result)

if __name__ == "__main__":
    app.run(debug=True)
