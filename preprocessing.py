from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "insurance_claims.csv"
    return pd.read_csv(data_path)


def preprocess() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = load_data()
    
    # Copy dataset after preprocessing
    data = df.copy()

    # Initialize encoder
    le = LabelEncoder()

    # Encode categorical columns
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

    X = data.drop('fraud_reported', axis=1)
    y = data['fraud_reported']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0
    )

    # Initialize scaler
    std_scaler = StandardScaler()

    # Fit on training data and transform
    X_train = std_scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X.columns)

    # Transform test data using same scaler
    X_test = std_scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
