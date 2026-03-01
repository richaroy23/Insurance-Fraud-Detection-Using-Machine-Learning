from flask import Flask, render_template, request
from typing import List
import numpy as np
import joblib
import os


# Load trained model, scaler, and feature list
model_path = "models/best_model.pkl"
scaler_path = "models/std_scaler.pkl"
features_path = "models/model_features.pkl"

# Check if artifacts exist
if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
    raise FileNotFoundError(
        "Model artifacts not found. Please run model_training.py first to train and save the model."
    )

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_features: List[str] = joblib.load(features_path)

print(f"✓ Model loaded successfully")
print(f"✓ Scaler loaded successfully")
print(f"✓ Features loaded: {len(model_features)} features")
print(f"  Features: {model_features}")

app = Flask(__name__)


@app.route('/')
def home():
    """Render home page with dynamic form fields based on model features."""
    return render_template("index.html", features=model_features)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the web form.
    
    Expects form data with keys matching model_features.
    Returns prediction result rendered in the template.
    """
    try:
        # Extract input values in exact feature order
        input_values = []
        for feature in model_features:
            if feature not in request.form:
                raise ValueError(f"Missing required field: {feature}")
            input_values.append(float(request.form[feature]))
        
        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)
        
        # Apply scaling (must match training preprocessing)
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Convert prediction to human-readable result
        if prediction == 1:
            result = "Fraud"
            confidence = probability[1] * 100
        else:
            result = "Not Fraud"
            confidence = probability[0] * 100
        
        prediction_text = f"Prediction: {result} (Confidence: {confidence:.1f}%)"
        
        return render_template(
            "index.html",
            features=model_features,
            prediction_text=prediction_text
        )
    
    except ValueError as ve:
        error_msg = f"Input Error: {str(ve)}"
        return render_template(
            "index.html",
            features=model_features,
            prediction_text=error_msg
        )
    
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        return render_template(
            "index.html",
            features=model_features,
            prediction_text=error_msg
        )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)