from flask import Flask, render_template, request
from typing import Dict, List, Any
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_data


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


def compute_feature_guidance(features: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compute min/max/example guidance from encoded training data for each model feature."""
    df = load_data()
    df = df.dropna(axis=1, how='all')

    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['policy_bind_year'] = df['policy_bind_date'].dt.year
    df['policy_bind_month'] = df['policy_bind_date'].dt.month
    df['policy_bind_day'] = df['policy_bind_date'].dt.day
    df.drop('policy_bind_date', axis=1, inplace=True)

    data = df.copy()
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'str':
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    feature_df = data.drop('fraud_reported', axis=1)

    guidance: Dict[str, Dict[str, Any]] = {}
    for feature in features:
        if feature in feature_df.columns:
            min_value = float(feature_df[feature].min())
            max_value = float(feature_df[feature].max())
            example_value = float(feature_df[feature].median())
            guidance[feature] = {
                "min": min_value,
                "max": max_value,
                "example": example_value,
                "label": feature.replace('_', ' ').title(),
            }

    return guidance


feature_guidance = compute_feature_guidance(model_features)


def get_hobbies_id_mapping() -> List[str]:
    """Return stable LabelEncoder mapping lines for insured_hobbies (id -> hobby)."""
    df = load_data()
    df = df.dropna(axis=1, how='all')

    if 'insured_hobbies' not in df.columns:
        return []

    hobby_encoder = LabelEncoder()
    hobby_encoder.fit(df['insured_hobbies'].astype(str))

    mapping_lines: List[str] = []
    for hobby_id, hobby_name in enumerate(hobby_encoder.classes_):
        mapping_lines.append(f"{hobby_id} → {hobby_name}")

    return mapping_lines


def get_feature_description(feature: str) -> str:
    """Return user-friendly meaning for each model feature."""
    descriptions = {
        "incident_severity": "Severity level of the incident (higher value means more severe incident).",
        "insured_hobbies": "Hobby code for the insured person. Use the ID-to-hobby mapping shown below.",
        "vehicle_claim": "Amount claimed for vehicle damage.",
        "insured_zip": "Encoded/normalized location indicator based on ZIP region.",
        "total_claim_amount": "Total amount claimed across all components.",
        "property_claim": "Amount claimed for property-related damage.",
        "incident_date": "Encoded incident date bucket used during model training.",
        "months_as_customer": "How long the person has been a customer (in months).",
    }
    return descriptions.get(feature, "Input feature used by the fraud detection model.")


def value_level(value: float, min_value: float, max_value: float) -> str:
    """Classify entered value level inside feature range."""
    if max_value <= min_value:
        return "typical"

    ratio = (value - min_value) / (max_value - min_value)
    if ratio < 0.33:
        return "low"
    if ratio < 0.66:
        return "medium"
    return "high"


def build_prediction_explanation(
    form_values: Dict[str, str],
    guidance: Dict[str, Dict[str, Any]],
    result: str,
    confidence: float,
) -> Dict[str, Any]:
    """Generate user-friendly explanation for prediction based on entered values."""
    insights: List[str] = []
    numeric_values: Dict[str, float] = {}

    for feature, raw_value in form_values.items():
        try:
            numeric_values[feature] = float(raw_value)
        except ValueError:
            continue

    for feature in model_features:
        if feature not in numeric_values or feature not in guidance:
            continue

        min_value = float(guidance[feature]["min"])
        max_value = float(guidance[feature]["max"])
        current_value = numeric_values[feature]
        level = value_level(current_value, min_value, max_value)
        label = guidance[feature]["label"]
        insights.append(
            f"{label}: entered {current_value:.2f}, which is {level} within typical range ({min_value:.2f} to {max_value:.2f})."
        )

    summary = (
        f"The model predicts '{result}' with {confidence:.1f}% confidence based on how your values compare to known training patterns."
    )

    if result == "Fraud":
        interpretation = (
            "This indicates your combination of values looks more similar to previously flagged suspicious claim patterns."
        )
    else:
        interpretation = (
            "This indicates your combination of values looks more similar to previously legitimate claim patterns."
        )

    return {
        "summary": summary,
        "interpretation": interpretation,
        "insights": insights,
    }


for feature_name in model_features:
    if feature_name in feature_guidance:
        feature_guidance[feature_name]["description"] = get_feature_description(feature_name)

if "insured_hobbies" in feature_guidance:
    feature_guidance["insured_hobbies"]["id_mapping"] = get_hobbies_id_mapping()

print(f"✓ Model loaded successfully")
print(f"✓ Scaler loaded successfully")
print(f"✓ Features loaded: {len(model_features)} features")
print(f"  Features: {model_features}")

app = Flask(__name__)


@app.route('/')
def home():
    """Render home page with dynamic form fields based on model features."""
    return render_template(
        "index.html",
        features=model_features,
        feature_guidance=feature_guidance,
        form_values={},
        prediction_explanation=None,
    )


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
        form_values: Dict[str, str] = {}
        for feature in model_features:
            if feature not in request.form:
                raise ValueError(f"Missing required field: {feature}")

            raw_value = request.form.get(feature, "").strip()
            if raw_value == "":
                raise ValueError(f"Please enter a value for: {feature}")

            try:
                parsed_value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Invalid number for {feature}: {raw_value}") from exc

            form_values[feature] = raw_value
            input_values.append(parsed_value)
        
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
        prediction_explanation = build_prediction_explanation(
            form_values=form_values,
            guidance=feature_guidance,
            result=result,
            confidence=confidence,
        )
        
        return render_template(
            "index.html",
            features=model_features,
            feature_guidance=feature_guidance,
            form_values=form_values,
            prediction_text=prediction_text,
            prediction_explanation=prediction_explanation,
        )
    
    except ValueError as ve:
        error_msg = f"Input Error: {str(ve)}"
        return render_template(
            "index.html",
            features=model_features,
            feature_guidance=feature_guidance,
            form_values=request.form.to_dict(),
            prediction_text=error_msg,
            prediction_explanation=None,
        )
    
    except Exception as e:
        error_msg = f"Prediction Error: {str(e)}"
        return render_template(
            "index.html",
            features=model_features,
            feature_guidance=feature_guidance,
            form_values=request.form.to_dict(),
            prediction_text=error_msg,
            prediction_explanation=None,
        )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)