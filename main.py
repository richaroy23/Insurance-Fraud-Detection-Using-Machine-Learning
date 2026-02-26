from src.data_loader import load_data
from src import eda
from src import preprocessing as prep
from src import visualization as viz


# PHASE 1 — BASIC EDA
def run_eda(df):
    print("\n================ PHASE 1: EDA ================\n")

    eda.dataset_overview(df)
    eda.missing_values(df)
    eda.statistical_summary(df)
    eda.target_distribution(df)

    eda.multiple_numerical_distributions(
        df,
        ["age", "total_claim_amount"]
    )


# PHASE 2 — OUTLIER HANDLING
def run_outlier_handling(df):
    print("\n================ PHASE 2: OUTLIER HANDLING ================\n")

    column = "policy_annual_premium"

    viz.boxplot(df, column)

    lower, upper = prep.compute_iqr_bounds(df, column)
    print("\nIQR Bounds")
    print("Lower Bound:", lower)
    print("Upper Bound:", upper)

    viz.distribution_plot(df, column)

    df = prep.log_transform(df, column)

    viz.distribution_plot(df, column)

    return df


# PHASE 3 — VISUAL ANALYSIS
def run_visual_analysis(df):
    print("\n================ PHASE 3: VISUAL ANALYSIS ================\n")

    eda.fraud_countplot(df)
    eda.incident_severity_pie(df)
    eda.age_distribution(df)


# PHASE 4 — MULTIVARIATE ANALYSIS
def run_multivariate_analysis(df):
    print("\n================ PHASE 4: MULTIVARIATE ANALYSIS ================\n")

    eda.correlation_heatmap(df)

    df, dropped_cols = prep.drop_correlated_features(df, threshold=0.9)

    print("Dropped columns:", dropped_cols)

    return df


# PHASE 5 — FEATURE ENGINEERING
def run_feature_engineering(df):
    print("\n================ PHASE 5: FEATURE ENGINEERING ================\n")

    df, encoders = prep.encode_categorical_features(df)

    X, y = prep.split_features_target(df)

    X_train, X_test, y_train, y_test = prep.split_train_test(X, y)

    return X_train, X_test, y_train, y_test, encoders


def run_feature_scaling(X_train, X_test):
    print("\n================ PHASE 6: FEATURE SCALING ================\n")

    X_train, X_test, scaler = prep.scale_features(X_train, X_test)

    return X_train, X_test, scaler


# MAIN PIPELINE
def main():

    print("\n========== INSURANCE FRAUD DETECTION PIPELINE ==========\n")

    # Load Data
    df = load_data()

    # Sequential Execution
    run_eda(df)
    df = run_outlier_handling(df)
    run_visual_analysis(df)
    df = run_multivariate_analysis(df)

    X_train, X_test, y_train, y_test, encoders = run_feature_engineering(df)
    X_train, X_test, scaler = run_feature_scaling(X_train, X_test)

    print("\nPipeline execution completed successfully.\n")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()