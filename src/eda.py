import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")


# Dataset Overview
def dataset_overview(df):
    print("\n========== DATASET OVERVIEW ==========\n")

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nInfo:")
    df.info()


# Missing Values
def missing_values(df):
    print("\n========== MISSING VALUES ==========\n")

    missing = df.isna().sum()
    print(missing)

    total_missing = missing.sum()
    print("\nTotal Missing Values:", total_missing)


# Statistical Summary
def statistical_summary(df):
    print("\n========== STATISTICAL SUMMARY ==========\n")
    print(df.describe())


# Target Variable Analysis
def target_distribution(df, target="fraud_reported"):
    print("\n========== TARGET DISTRIBUTION ==========\n")

    counts = df[target].value_counts()
    percentage = df[target].value_counts(normalize=True) * 100

    print("Counts:\n", counts)
    print("\nPercentage:\n", percentage)

    plt.figure(figsize=(6,4))
    sns.countplot(x=target, data=df)
    plt.title("Fraud Distribution")
    plt.show()



# Numerical Feature Distributions
def numerical_distribution(df, column):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def multiple_numerical_distributions(df, columns):
    for col in columns:
        numerical_distribution(df, col)

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True),
                cmap="coolwarm",
                annot=False)
    plt.title("Correlation Heatmap")
    plt.show()
eda.correlation_heatmap(df)

# Fraud Countplot
def fraud_countplot(df, column="fraud_reported"):
    plt.figure(figsize=(6,4))
    sns.countplot(x=column, data=df)
    plt.title("Fraud Reported Count")
    plt.xlabel("Fraud Reported")
    plt.ylabel("Count")
    plt.show()

    total = len(df)
    fraud_count = df[column].value_counts().get("Y", 0)

    print(f"\nFraud cases: {fraud_count}")
    print(f"Total claims: {total}")

# Incident Severity Pie Chart
def incident_severity_pie(df, column="incident_severity"):

    counts = df[column].value_counts()

    plt.figure(figsize=(7,7))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Damage Visualization")
    plt.show()

# Age Distribution Histogram
def age_distribution(df, column="age"):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], bins=12, kde=True, color="salmon")

    plt.title("Age Distribution", fontsize=18)
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Number of People", fontsize=14)

    plt.show()

# Correlation Heatmap
def correlation_heatmap(df):

    plt.figure(figsize=(14,10))
    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=True,
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.show()