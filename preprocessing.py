from pathlib import Path
from typing import Optional, List, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def load_data() -> pd.DataFrame:
    """Load insurance claims dataset from CSV file.
    
    Returns:
        pd.DataFrame: Raw insurance claims data
    """
    data_path = Path(__file__).resolve().parent / "insurance_claims.csv"
    return pd.read_csv(data_path)


def preprocess(selected_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """
    Preprocess insurance claims data with optional feature selection.
    
    Pipeline order:
    1. Load data
    2. Clean (remove null columns)
    3. Feature engineering (datetime parsing)
    4. Encode categorical variables
    5. Select features (if specified)
    6. Train-test split
    7. Scale features
    
    Args:
        selected_features: Optional list of feature names to use.
                          If None, all features are used.
    
    Returns:
        Tuple containing:
        - X_train: Scaled training features (numpy array)
        - X_test: Scaled test features (numpy array)
        - y_train: Training labels
        - y_test: Test labels
        - scaler: Fitted StandardScaler instance
    """
    # Load data
    df = load_data()
    
    # Remove junk columns with all null values
    df = df.dropna(axis=1, how='all')
    
    # Feature engineering: Parse policy_bind_date
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['policy_bind_year'] = df['policy_bind_date'].dt.year
    df['policy_bind_month'] = df['policy_bind_date'].dt.month
    df['policy_bind_day'] = df['policy_bind_date'].dt.day
    df.drop('policy_bind_date', axis=1, inplace=True)
    
    # Copy dataset after preprocessing
    data = df.copy()

    # Initialize encoders dictionary
    encoders = {}

    # Encode categorical columns
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype == 'str':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

    # Separate features and target
    X = data.drop('fraud_reported', axis=1)
    y = data['fraud_reported']
    
    # Feature selection: Use only specified features if provided
    if selected_features is not None:
        # Ensure all selected features exist in the dataset
        missing_features = set(selected_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Selected features not found in dataset: {missing_features}")
        X = X[selected_features]
        print(f"Using {len(selected_features)} selected features: {selected_features}")
    else:
        print(f"Using all {len(X.columns)} features")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Initialize and fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
