from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import joblib
import os

from evaluation import evaluate_classification
from preprocessing import preprocess


def main() -> None:
    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(
            random_state=0,
            class_weight='balanced'
        ),
        "KNN": KNeighborsClassifier(n_neighbors=30, weights='distance'),
        "Logistic Regression": LogisticRegressionCV(
            solver='lbfgs',
            max_iter=5000,
            cv=10,
            class_weight='balanced'
        ),
        "Naïve Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }
    cv = StratifiedKFold(
            n_splits=5, 
            shuffle=True, 
            random_state=42
        )
    
    results = []

    for name, model in models.items():
        print("\n" + "=" * 50)
        print(f"Training {name}...")
        print("=" * 50)

        model.fit(X_train, y_train)
    
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring='f1'
        )

        print(f"\n{name}")
        print("Mean CV F1-score:", scores.mean())
        print("CV Std:", scores.std())

        evaluate_classification(model, X_test, y_test)
        
        results.append({
            "Model": name,
            "CV Mean F1-score": scores.mean(),
            "CV Std": scores.std()
        })

#   Model Comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="CV Mean F1-score",
        ascending=False
    )

    print("\nModel Comparison:")
    print(results_df)

#   Select the best model
    best_model_name = results_df.iloc[0]["Model"]
    print("\nBest Model:", best_model_name)

    best_model = models[best_model_name]
#   Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\nBest model '{best_model_name}' saved to 'models/best_model.pkl'.")

if __name__ == "__main__":
    main()