# Insurance Fraud Detection Using Machine Learning

A production-ready ML system that predicts whether an insurance claim is **fraudulent or legitimate** using Random Forest classification with optimized feature selection.

---

## 🎯 Overview

This system analyzes insurance claims data to detect fraudulent activities across automobile, health, and property insurance. The project has been refactored for production deployment with:

✅ **Feature Selection** - Reduced from 40 to 8 most important features  
✅ **High Accuracy** - 84% accuracy with 0.70 F1-score  
✅ **Modular Code** - Clean architecture with type hints and documentation  
✅ **Web Interface** - Dynamic Flask app with responsive UI  
✅ **Reproducible** - Fixed random seeds and consistent preprocessing  

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **84%** |
| **F1 Score** | **0.7037** |
| **Precision (Fraud)** | 0.64 |
| **Recall (Fraud)** | 0.78 |
| **Precision (Not Fraud)** | 0.92 |
| **Recall (Not Fraud)** | 0.86 |

**Confusion Matrix:**
```
                Predicted
              Not Fraud  Fraud
Actual Not      130       21
       Fraud     11       38
```

🎯 **The model correctly identifies 78% of actual fraud cases while maintaining 84% overall accuracy.**

---

## 🔑 8 Selected Features (by importance)

The model uses only 8 optimized features selected via Random Forest feature importance:

1. **incident_severity** (15.76%) - Severity of the incident
2. **insured_hobbies** (8.07%) - Hobbies of insured person
3. **vehicle_claim** (4.47%) - Vehicle claim amount
4. **insured_zip** (3.92%) - Zip code of insured
5. **total_claim_amount** (3.63%) - Total claim amount
6. **property_claim** (3.52%) - Property claim amount
7. **incident_date** (3.40%) - Date of incident (encoded)
8. **months_as_customer** (3.20%) - Duration as customer

---

## 🚀 Quick Start

### 1. Train the Model
```bash
python model_training.py
```
This trains the model, selects top features, and saves artifacts to `models/` directory.

### 2. Run the Web Application
```bash
python app.py
```
Open browser to `http://localhost:5000`

### 3. Make Predictions
Enter values for the 8 features and click "Analyze Claim" to get fraud prediction with confidence score.

---

## 📁 Project Structure

```
├── preprocessing.py          # Data preprocessing with feature selection
├── model_training.py         # Feature importance & model training
├── app.py                    # Flask web application
├── templates/index.html      # Dynamic web form
├── models/                   # Saved model artifacts
│   ├── best_model.pkl       # Trained RandomForest model
│   ├── std_scaler.pkl       # StandardScaler
│   └── model_features.pkl   # Selected feature names
└── insurance_claims.csv      # Dataset
```

---

## 🛠️ Technologies Used

- **Python 3.12**
- **scikit-learn** - RandomForest, preprocessing, metrics
- **Flask** - Web framework
- **Pandas & NumPy** - Data manipulation
- **Joblib** - Model serialization

---

## ✨ Key Features

### 🔍 Intelligent Feature Selection
- Analyzes 40+ features and selects top 8 most important
- Reduces complexity while maintaining high accuracy
- Makes deployment practical and user-friendly

### 📈 Production-Ready Pipeline
- **No data leakage** - Proper train-test split before scaling
- **Reproducible** - Fixed random seeds throughout
- **Type hints** - Full type annotations for maintainability
- **Error handling** - Comprehensive validation

### 🎨 Modern Web Interface
- Dynamic form generation based on selected features
- Real-time validation
- Confidence scores with predictions
- Responsive design for mobile and desktop

---

## 📝 Model Training Pipeline

```python
# 1. Feature Importance Analysis
top_features = get_top_features(n_features=8)

# 2. Train Final Model
model, scaler, f1 = train_final_model(top_features)

# 3. Save Artifacts
save_artifacts(model, scaler, top_features)
```

Pipeline order ensures:
1. Load → Clean → Encode → Select Features → Split → Scale
2. No data leakage (scaling after train-test split)
3. Consistent feature order for predictions

---

## ⚙️ Configuration

To change the number of features, edit `model_training.py`:

```python
N_FEATURES = 8  # Change to 5, 10, 15, etc.
```

Then retrain:
```bash
python model_training.py
```

The web form automatically updates to match!

---

## 🧪 Validation

Run the validation script to check everything:

```bash
python validate_setup.py
```

Checks:
- ✓ Project files present
- ✓ Required packages installed
- ✓ Model artifacts saved correctly
- ✓ Prediction pipeline working

---

## 📊 Use Cases

### 🚗 Automobile Insurance
- Detects staged accidents and exaggerated claims
- Analyzes damage patterns and claim history

### 🏥 Health Insurance
- Identifies overbilling and duplicate claims
- Flags unnecessary medical procedures

### 🏠 Property Insurance
- Detects inflated property values
- Identifies suspicious claim timing

---

## 🎓 Best Practices Implemented

✅ Modular architecture with reusable functions  
✅ Comprehensive documentation and type hints  
✅ No hardcoded feature names  
✅ Consistent preprocessing pipeline  
✅ Proper artifact management  
✅ Dynamic UI adapts to model changes  
✅ Error handling and validation  

---

## 🚀 Benefits

- **84% accuracy** with only 8 features (down from 40+)
- **Faster deployment** - Simpler data collection
- **Lower costs** - Fewer features to maintain
- **Better UX** - Quick form completion
- **Maintainable** - Clean, documented code

