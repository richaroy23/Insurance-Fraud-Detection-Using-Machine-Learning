from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd
import numpy as np
import joblib
import os

from preprocessing import preprocess, load_data


def get_top_features(n_features: int = 8, random_state: int = 42) -> List[str]:
    """
    Train RandomForest on full feature set and extract top N important features.
    
    Args:
        n_features: Number of top features to select (default: 8)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        List of top N feature names sorted by importance (descending)
    """
    print("=" * 60)
    print("STEP 1: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Load and preprocess with ALL features
    X_train, X_test, y_train, y_test, scaler = preprocess(selected_features=None)
    
    # Get feature names from original data (before scaling)
    df = load_data()
    df = df.dropna(axis=1, how='all')
    
    # Parse dates
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['policy_bind_year'] = df['policy_bind_date'].dt.year
    df['policy_bind_month'] = df['policy_bind_date'].dt.month
    df['policy_bind_day'] = df['policy_bind_date'].dt.day
    df.drop('policy_bind_date', axis=1, inplace=True)
    
    # Encode categorical
    from sklearn.preprocessing import LabelEncoder
    data = df.copy()
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'str':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    
    X = data.drop('fraud_reported', axis=1)
    feature_names = X.columns.tolist()
    
    print(f"\nTotal features available: {len(feature_names)}")
    
    # Train RandomForest for feature importance
    print("\nTraining RandomForest on full feature set...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(n_features)['feature'].tolist()
    
    print("\n" + "=" * 60)
    print(f"TOP {n_features} MOST IMPORTANT FEATURES:")
    print("=" * 60)
    for idx, row in feature_importance_df.head(n_features).iterrows():
        print(f"{row['feature']:.<40} {row['importance']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Selected features: {top_features}")
    print("=" * 60)
    
    return top_features


def train_final_model(
    selected_features: List[str],
    random_state: int = 42
) -> Tuple[RandomForestClassifier, object, float]:
    """
    Train final RandomForest model using only selected features.
    
    Args:
        selected_features: List of feature names to use for training
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (trained_model, scaler, f1_score)
    """
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING FINAL MODEL")
    print("=" * 60)
    
    # Preprocess with selected features only
    X_train, X_test, y_train, y_test, scaler = preprocess(selected_features=selected_features)
    
    print(f"\nTraining RandomForest with {len(selected_features)} features...")
    
    # Train final model
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"\nF1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return final_model, scaler, f1


def save_artifacts(
    model: RandomForestClassifier,
    scaler: object,
    model_features: List[str],
    save_dir: str = "models"
) -> None:
    """
    Save trained model, scaler, and feature list to disk.
    
    Args:
        model: Trained RandomForestClassifier
        scaler: Fitted StandardScaler
        model_features: Ordered list of feature names used in training
        save_dir: Directory to save artifacts (default: 'models')
    """
    print("\n" + "=" * 60)
    print("STEP 3: SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(save_dir, "std_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save feature names (in exact order)
    features_path = os.path.join(save_dir, "model_features.pkl")
    joblib.dump(model_features, features_path)
    print(f"✓ Features saved to: {features_path}")
    
    print("\n" + "=" * 60)
    print("All artifacts saved successfully!")
    print("=" * 60)


def main() -> None:
    """
    Main pipeline for feature selection, model training, and saving artifacts.
    """
    print("\n" + "=" * 70)
    print(" INSURANCE FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Configuration
    N_FEATURES = 8  # Number of top features to select
    RANDOM_STATE = 42  # For reproducibility
    
    # Step 1: Get top features based on importance
    top_features = get_top_features(n_features=N_FEATURES, random_state=RANDOM_STATE)
    
    # Step 2: Train final model with selected features
    final_model, scaler, f1_score = train_final_model(
        selected_features=top_features,
        random_state=RANDOM_STATE
    )
    
    # Step 3: Save all artifacts
    save_artifacts(
        model=final_model,
        scaler=scaler,
        model_features=top_features
    )
    
    print("\n" + "=" * 70)
    print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal Model Performance:")
    print(f"  • Features used: {N_FEATURES}")
    print(f"  • F1 Score: {f1_score:.4f}")
    print(f"\nModel artifacts saved in 'models/' directory.")
    print(f"Selected features: {top_features}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()