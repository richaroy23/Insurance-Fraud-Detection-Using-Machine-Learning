"""
Enhanced test script to verify the trained model with comprehensive metrics
"""

import joblib
import pandas as pd
import numpy as np
from preprocessing import preprocess
from sklearn.metrics import (confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, auc)
import os
from datetime import datetime

def test_model():
    """Test the trained model on test data with comprehensive metrics"""
    
    # Load the trained model and scaler
    print("Loading model and scaler...")
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/std_scaler.pkl")
    
    # Load preprocessed data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _, _ = preprocess()
    
    print("\n" + "="*70)
    print("INSURANCE FRAUD DETECTION MODEL - COMPREHENSIVE TEST REPORT")
    print("="*70)
    
    # Test on multiple samples
    num_samples = 10
    print(f"\nTesting on {num_samples} random samples from test data:\n")
    
    for i in range(min(num_samples, len(X_test))):
        sample = X_test.iloc[i:i+1]
        actual = y_test.iloc[i]
        
        # Make prediction
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]
        
        # Display results
        print(f"Sample {i+1}:")
        print(f"  Actual:     {'FRAUD ❌' if actual == 1 else 'LEGITIMATE ✓'}")
        print(f"  Predicted:  {'FRAUD ❌' if prediction == 1 else 'LEGITIMATE ✓'}")
        print(f"  Confidence: {max(probability)*100:.2f}%")
        print(f"  Match: {'✓ CORRECT' if prediction == actual else '✗ INCORRECT'}")
        print()
    
    # Get predictions for all test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    # ========== CONFUSION MATRIX ==========
    print("="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{'':20} Predicted Legitimate  Predicted Fraud")
    print(f"{'Actual Legitimate':20} {tn:15} {fp:15}")
    print(f"{'Actual Fraud':20} {fn:15} {tp:15}")
    
    # ========== DETAILED METRICS ==========
    print("\n" + "="*70)
    print("DETAILED PERFORMANCE METRICS")
    print("="*70)
    
    accuracy = (predictions == y_test).sum() / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False negative rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print(f"\n{'Metric':<30} {'Value':<15} {'Description':<40}")
    print("-" * 85)
    print(f"{'Accuracy':<30} {accuracy*100:<14.2f}% {'Correct predictions / Total'}")
    print(f"{'Precision (Fraud)':<30} {precision*100:<14.2f}% {'TP / (TP + FP)'}")
    print(f"{'Recall (Fraud)':<30} {recall*100:<14.2f}% {'TP / (TP + FN)'}")
    print(f"{'F1-Score':<30} {f1:<14.4f} {'Harmonic mean of Precision & Recall'}")
    print(f"{'Specificity':<30} {specificity*100:<14.2f}% {'TN / (TN + FP)'}")
    print(f"{'False Positive Rate':<30} {fpr*100:<14.2f}% {'FP / (FP + TN)'}")
    print(f"{'False Negative Rate':<30} {fnr*100:<14.2f}% {'FN / (FN + TP)'}")
    print(f"{'ROC-AUC Score':<30} {roc_auc:<14.4f} {'Area Under ROC Curve'}")
    
    # ========== CLASSIFICATION REPORT ==========
    print("\n" + "="*70)
    print("SKLEARN CLASSIFICATION REPORT")
    print("="*70)
    print("\n" + classification_report(y_test, predictions, 
                                       target_names=['Legitimate', 'Fraud']))
    
    # ========== SUMMARY STATISTICS ==========
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    fraud_total = (y_test == 1).sum()
    legitimate_total = (y_test == 0).sum()
    
    print(f"\nTotal Test Samples: {len(y_test)}")
    print(f"  - Legitimate Claims: {legitimate_total} ({legitimate_total/len(y_test)*100:.1f}%)")
    print(f"  - Fraudulent Claims: {fraud_total} ({fraud_total/len(y_test)*100:.1f}%)")
    
    print(f"\nModel Predictions:")
    print(f"  - Predicted Legitimate: {(predictions == 0).sum()}")
    print(f"  - Predicted Fraud: {(predictions == 1).sum()}")
    
    print(f"\nDetection Performance:")
    print(f"  - Fraud Detected: {tp} out of {fraud_total} ({recall*100:.2f}%)")
    print(f"  - Frauds Missed: {fn}")
    print(f"  - False Alarms (Legitimate marked as Fraud): {fp}")
    
    # ========== SAVE RESULTS TO CSV ==========
    print("\n" + "="*70)
    print("SAVING DETAILED RESULTS")
    print("="*70)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions,
        'Fraud_Probability': probabilities,
        'Correct': (predictions == y_test.values)
    })
    
    # Save to CSV
    os.makedirs('test_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'test_results/model_predictions_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed predictions saved to: {csv_path}")
    
    # Save summary report
    summary_path = f'test_results/model_report_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("INSURANCE FRAUD DETECTION MODEL - TEST REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Precision (Fraud): {precision*100:.2f}%\n")
        f.write(f"Recall (Fraud): {recall*100:.2f}%\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n")
    
    print(f"✓ Summary report saved to: {summary_path}")
    
    print("\n" + "="*70)
    print("✓ MODEL TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    test_model()
