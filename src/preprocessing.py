import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# OUTLIER HANDLING
def compute_iqr_bounds(df, column):
    """
    Compute lower and upper bounds for outlier detection using IQR method.
    """
    if column not in df.columns:
        raise ValueError(f"{column} not found in dataframe")

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper


def log_transform(df, column):
    """
    Apply log transformation to reduce skewness.
    Handles zero or positive values safely.
    """
    if (df[column] < 0).any():
        raise ValueError(f"{column} contains negative values — cannot apply log transform")

    df[column] = np.log1p(df[column])
    print(f"{column} log transformed")

    return df


# MULTICOLLINEARITY HANDLING
def find_highly_correlated_features(df, threshold=0.9, exclude=None):
    """
    Identify highly correlated features based on correlation threshold.
    """
    if exclude is None:
        exclude = []

    corr_matrix = df.corr(numeric_only=True).abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold) and column not in exclude
    ]

    return to_drop


def drop_correlated_features(df, threshold=0.9, target="fraud_reported"):
    """
    Drop highly correlated features but keep target variable safe.
    """
    to_drop = find_highly_correlated_features(
        df,
        threshold=threshold,
        exclude=[target]
    )

    print("\nHighly correlated features to drop:")
    print(to_drop)

    df = df.drop(columns=to_drop)

    print("\nRemaining features:", df.shape[1])

    return df, to_drop



# CATEGORICAL ENCODING
def encode_categorical_features(df):
    """
    Encode all object type columns using Label Encoding.
    Returns encoded dataframe and dictionary of encoders.
    """
    print("\nEncoding categorical features...")

    label_encoders = {}

    for column in df.columns:
        if df[column].dtype == "object":

            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))

            label_encoders[column] = le
            print(f"Encoded: {column}")

    return df, label_encoders


# FEATURE / TARGET SPLIT
def split_features_target(df, target="fraud_reported"):
    """
    Separate features and target variable.
    """
    if target not in df.columns:
        raise ValueError(f"{target} not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target]

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


# TRAIN TEST SPLIT
def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    Uses stratification for class imbalance.
    """
    print("\nPerforming train-test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("Training size:", X_train.shape)
    print("Testing size:", X_test.shape)

    return X_train, X_test, y_train, y_test

# FEATURE SCALING
def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    Fit on training data only to avoid data leakage.
    """

    print("\nApplying Standard Scaling...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame (keeps column names)
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    print("Scaling completed")

    return X_train_scaled, X_test_scaled, scaler