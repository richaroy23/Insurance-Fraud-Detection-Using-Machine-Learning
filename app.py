from flask import Flask, render_template, request
import numpy as np
import joblib


# Load trained model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/std_scaler.pkl")   

app = Flask(__name__)


# Home page route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get all input values from form
        input_values = [float(x) for x in request.form.values()]

        # Convert to numpy array
        input_array = np.array(input_values).reshape(1, -1)

        # Apply scaling (if used in training)
        input_scaled = scaler.transform(input_array)

        # Model prediction
        prediction = model.predict(input_scaled)[0]

        # Convert prediction to readable text
        if prediction == 1:
            result = "Fraud Insurance Claim"
        else:
            result = "Legal Insurance Claim"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))


# Run server
if __name__ == "__main__":
    app.run(debug=True)