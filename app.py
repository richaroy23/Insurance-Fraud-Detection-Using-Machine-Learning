from flask import Flask, render_template, request, jsonify
from typing import Dict, List, Any, Mapping, Tuple
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


def build_incident_date_encoder() -> LabelEncoder:
    """Build LabelEncoder for incident_date using the same logic as training."""
    df = load_data()
    df = df.dropna(axis=1, how='all')

    if 'incident_date' not in df.columns:
        raise ValueError("incident_date column not found in dataset")

    encoder = LabelEncoder()
    encoder.fit(df['incident_date'].astype(str))
    return encoder


incident_date_encoder = build_incident_date_encoder()
incident_date_known_dates = set(incident_date_encoder.classes_)
incident_date_min = str(min(incident_date_encoder.classes_))
incident_date_max = str(max(incident_date_encoder.classes_))
incident_date_example = str(incident_date_encoder.classes_[len(incident_date_encoder.classes_) // 2])


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


def encode_incident_date(raw_value: str) -> float:
    """Convert user-provided date to model incident_date encoding.
    
    Accepts DD-MM-YYYY or YYYY-MM-DD format.
    Encoding behavior is aligned with training preprocessing:
    - If exact YYYY-MM-DD date exists in training classes, use it directly.
    - Otherwise, map by month/day to year 2015 and validate that mapped date exists.
    """
    value = str(raw_value).strip()
    if value == "":
        raise ValueError("Please enter a value for: incident_date")

    parsed_date = None
    
    # Try DD-MM-YYYY format first
    parsed_date = pd.to_datetime(value, format='%d-%m-%Y', errors='coerce')
    
    # If that fails, try YYYY-MM-DD format (backward compatibility)
    if pd.isna(parsed_date):
        parsed_date = pd.to_datetime(value, format='%Y-%m-%d', errors='coerce')
    
    if pd.isna(parsed_date):
        raise ValueError(
            "Invalid date format. Please use DD-MM-YYYY (e.g., 15-06-2026) or YYYY-MM-DD."
        )

    exact_normalized = parsed_date.strftime('%Y-%m-%d')
    if exact_normalized in incident_date_known_dates:
        return float(incident_date_encoder.transform([exact_normalized])[0])

    mapped_2015 = parsed_date.replace(year=2015).strftime('%Y-%m-%d')
    if mapped_2015 not in incident_date_known_dates:
        raise ValueError(
            f"Date out of supported range. Please use dates between {incident_date_min} and {incident_date_max}, "
            "or keep month/day within that range."
        )

    normalized = mapped_2015
    return float(incident_date_encoder.transform([normalized])[0])


def parse_inputs_from_source(source: Mapping[str, Any]) -> Tuple[Dict[str, str], List[float]]:
    """Parse and validate inputs in model feature order from form or JSON source."""
    input_values: List[float] = []
    form_values: Dict[str, str] = {}

    for feature in model_features:
        if feature == 'incident_date':
            raw_incident_date = str(
                source.get('incident_date_actual', source.get('incident_date', ''))
            ).strip()
            encoded_incident_date = encode_incident_date(raw_incident_date)
            form_values[feature] = str(encoded_incident_date)
            input_values.append(encoded_incident_date)
            continue

        raw_value = str(source.get(feature, "")).strip()
        if raw_value == "":
            raise ValueError(f"Please enter a value for: {feature}")

        try:
            parsed_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid number for {feature}: {raw_value}") from exc

        form_values[feature] = raw_value
        input_values.append(parsed_value)

    return form_values, input_values


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
        incident_date_min=incident_date_min,
        incident_date_max=incident_date_max,
        incident_date_example=incident_date_example,
        form_values={},
        prediction_explanation=None,
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions. Returns JSON with prediction and insights.
    Expects JSON body with model feature keys.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract input values in exact feature order
        form_values, input_values = parse_inputs_from_source(data)
        
        # Convert to DataFrame with exact feature names/order
        input_frame = pd.DataFrame([input_values], columns=model_features)

        # Apply scaling
        input_scaled = scaler.transform(input_frame)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Convert prediction to human-readable result
        if prediction == 1:
            result = "Fraud"
            confidence = float(probability[1] * 100)
        else:
            result = "Not Fraud"
            confidence = float(probability[0] * 100)
        
        # Build explanation
        prediction_explanation = build_prediction_explanation(
            form_values=form_values,
            guidance=feature_guidance,
            result=result,
            confidence=confidence,
        )
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': round(confidence, 1),
            'summary': prediction_explanation['summary'],
            'interpretation': prediction_explanation['interpretation'],
            'insights': prediction_explanation['insights'],
        })
    
    except ValueError as ve:
        return jsonify({'error': f'Input Error: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction Error: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the web form.
    
    Expects form data with keys matching model_features.
    Returns prediction result rendered in the template.
    """
    try:
        # Extract input values in exact feature order
        form_values, input_values = parse_inputs_from_source(request.form)
        
        # Convert to DataFrame with exact feature names/order
        input_frame = pd.DataFrame([input_values], columns=model_features)

        # Apply scaling (must match training preprocessing)
        input_scaled = scaler.transform(input_frame)
        
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
            incident_date_min=incident_date_min,
            incident_date_max=incident_date_max,
            incident_date_example=incident_date_example,
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
            incident_date_min=incident_date_min,
            incident_date_max=incident_date_max,
            incident_date_example=incident_date_example,
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
            incident_date_min=incident_date_min,
            incident_date_max=incident_date_max,
            incident_date_example=incident_date_example,
            form_values=request.form.to_dict(),
            prediction_text=error_msg,
            prediction_explanation=None,
        )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)