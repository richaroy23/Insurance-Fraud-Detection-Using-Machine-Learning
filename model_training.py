from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV

from preprocessing import preprocess


def main() -> None:
    X_train, X_test, y_train, y_test = preprocess()
    
    # Initialize model
    dtc = DecisionTreeClassifier()

    # Train model
    dtc.fit(X_train, y_train)

    # Predict on test data
    y_pred = dtc.predict(X_test)

    # Training accuracy
    dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))

    # Testing accuracy
    dtc_test_acc = accuracy_score(y_test, y_pred)

    print("Decision Tree Training Accuracy:", dtc_train_acc)
    print("Decision Tree Testing Accuracy:", dtc_test_acc)

    # Initialize Random Forest model
    rf = RandomForestClassifier(random_state=0)

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on test data
    y_pred_rf = rf.predict(X_test)

    # Training accuracy
    rf_train_acc = accuracy_score(y_train, rf.predict(X_train))

    # Testing accuracy
    rf_test_acc = accuracy_score(y_test, y_pred_rf)

    print("Random Forest Training Accuracy:", rf_train_acc)
    print("Random Forest Testing Accuracy:", rf_test_acc)

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=30)

    # Train the model
    knn.fit(X_train, y_train)

    # Predict on test data
    y_pred_knn = knn.predict(X_test)

    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_knn))

    # Initialize model with cross-validation
    lr = LogisticRegressionCV(
        solver='lbfgs',
        max_iter=5000,
        cv=10
    )

    # Train model
    lr.fit(X_train, y_train)

    # Predict on test data
    y_pred_lr = lr.predict(X_test)

    # Evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))

    print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))

    # Initialize model
    gnb = GaussianNB()

    # Train model
    gnb.fit(X_train, y_train)

    # Predict on test data
    y_pred_gnb = gnb.predict(X_test)

    # Training accuracy
    gnb_train_acc = accuracy_score(y_train, gnb.predict(X_train))

    # Testing accuracy
    gnb_test_acc = accuracy_score(y_test, y_pred_gnb)

    print("Naïve Bayes Training Accuracy:", gnb_train_acc)
    print("Naïve Bayes Testing Accuracy:", gnb_test_acc)

    # Initialize SVM model
    svc = SVC()

    # Train model
    svc.fit(X_train, y_train)

    # Predict on test data
    y_pred_svc = svc.predict(X_test)

    # Accuracy
    svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
    svc_test_acc = accuracy_score(y_test, y_pred_svc)

    print("Training accuracy of SVM:", svc_train_acc)
    print("Testing accuracy of SVM:", svc_test_acc)

    # Evaluation metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svc))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_svc))


if __name__ == "__main__":
    main()