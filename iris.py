from flask import Flask, request, render_template_string
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# -------------------
# Train model
# -------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------
# Flask App
# -------------------
app = Flask(__name__)

# HTML form (inline, no extra files needed)
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Predictor</title>
</head>
<body>
    <h2>Iris Flower Prediction</h2>
    <form method="POST" action="/predict">
        <label>Sepal Length:</label>
        <input type="number" step="0.1" name="sepal_length" required><br><br>

        <label>Sepal Width:</label>
        <input type="number" step="0.1" name="sepal_width" required><br><br>

        <label>Petal Length:</label>
        <input type="number" step="0.1" name="petal_length" required><br><br>

        <label>Petal Width:</label>
        <input type="number" step="0.1" name="petal_width" required><br><br>

        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

# Result page
result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h2>Prediction Result</h2>
    <p>The flower is: <b>{{ flower }}</b></p>
    <a href="/">Go Back</a>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(form_html)

@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    flower = iris.target_names[prediction]

    return render_template_string(result_html, flower=flower)

if __name__ == "__main__":
    app.run(debug=True)
