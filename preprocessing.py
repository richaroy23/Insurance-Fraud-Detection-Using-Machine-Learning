from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "insurance_claims.csv"
    return pd.read_csv(data_path)


def preprocess() -> tuple:
    df = load_data()
    
    # Remove junk columns with all null values
    df = df.dropna(axis=1, how='all')
    
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

    X = data.drop('fraud_reported', axis=1)
    y = data['fraud_reported']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data and transform
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X.columns)

    # Transform test data using same scaler
    X_test =    scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    return X_train, X_test, y_train, y_test, scaler, encoders


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
