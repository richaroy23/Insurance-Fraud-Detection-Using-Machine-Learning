from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
import joblib

from preprocessing import preprocess


def make_predictions() -> None:
    # Load selected model features and get preprocessed data
    model_features = joblib.load("models/model_features.pkl")
    X_train, X_test, y_train, y_test, _ = preprocess(selected_features=model_features)
    
    # Initialize and train models
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(X_train, y_train)
    
    lr = LogisticRegressionCV(
        solver='lbfgs',
        max_iter=5000,
        cv=10,
        l1_ratios=(0.0,),
        use_legacy_attributes=False,
    )
    lr.fit(X_train, y_train)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    
    # Use one already-scaled test sample to keep feature dimensions consistent
    sample_scaled = X_test[0:1]
    actual_label = y_test.iloc[0]
    
    # Make predictions
    print("Decision Tree:", dtc.predict(sample_scaled))
    print("Random Forest:", rf.predict(sample_scaled))
    print("KNN:", knn.predict(sample_scaled))
    print("Logistic Regression:", lr.predict(sample_scaled))
    print("Naïve Bayes:", gnb.predict(sample_scaled))
    print("SVM:", svc.predict(sample_scaled))
    print("Actual Label:", actual_label)


if __name__ == "__main__":
    make_predictions()