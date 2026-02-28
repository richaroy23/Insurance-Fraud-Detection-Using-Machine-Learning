from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "insurance_claims.csv"
    return pd.read_csv(data_path)


def main() -> None:
    df = load_data()

    # Univariate analysis
    # Fraud Reported — Countplot
    sns.countplot(data=df, x="fraud_reported")
    plt.title("Fraud Reported Count")
    plt.show()

    # Incident Severity — Pie Chart
    df["incident_severity"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        figsize=(6, 6),
    )
    plt.title("Damage Visualization")
    plt.ylabel("")
    plt.show()

    # Age Distribution — Histogram
    sns.histplot(df["age"], bins=15, kde=True)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of People")
    plt.show()

    # Multivariate analysis
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Dropping Highly Correlated Features
    df = df.drop(
        [
            "months_as_customer",
            "injury_claim",
            "property_claim",
            "vehicle_claim",
        ],
        axis=1,
    )


if __name__ == "__main__":
    main()