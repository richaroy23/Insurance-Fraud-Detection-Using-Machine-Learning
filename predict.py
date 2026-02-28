from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import numpy as np

from preprocessing import preprocess


def make_predictions() -> None:
    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess()
    
    # Initialize and train models
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(X_train, y_train)
    
    lr = LogisticRegressionCV(solver='lbfgs', max_iter=5000, cv=10)
    lr.fit(X_train, y_train)
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    
    # Initialize and fit scaler
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    
    # Sample prediction data
    sample = [[328, 521585, 2012, 12, 250, 1000, 1406.91, 5600, 1, 100,
               25, 25, 50000, 0, 120, 23, 56, 52, 1, 123, 2, 3, 1, 0, 2,
               1, 150000, 2, 25, 2002]]
    
    # Scale the sample
    sample_array = np.array(sample)
    sample_scaled = std_scaler.transform(sample_array)
    
    # Make predictions
    print("Decision Tree:", dtc.predict(sample_scaled))
    print("Random Forest:", rf.predict(sample_scaled))
    print("KNN:", knn.predict(sample_scaled))
    print("Logistic Regression:", lr.predict(sample_scaled))
    print("Naïve Bayes:", gnb.predict(sample_scaled))
    print("SVM:", svc.predict(sample_scaled))


if __name__ == "__main__":
    make_predictions()