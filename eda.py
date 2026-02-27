import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import pickle
from scipy import stats
warnings.filterwarnings ('ignore')
plt.style.use('fivethirtyeight')


# Read the dataset
df = pd.read_csv("insurance_claims.csv")
df.head()
df.isna().any()
df.isna().sum()

# Visualizing Outliers with Boxplot
sns.boxplot(data=df, x='policy_annual_premium')
plt.title("Boxplot of Policy Annual Premium")
plt.show()

# Calculating IQR and Bounds
Q1 = df['policy_annual_premium'].quantile(0.25)
Q3 = df['policy_annual_premium'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Function to Visualize Distribution Before & After Transformation
def plot_distribution(feature):
    
    # Original distribution
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    sns.histplot(df[feature], kde=True)
    plt.title("Original Distribution")
    
    plt.subplot(1,2,2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.title("Original Probability Plot")
    
    plt.show()
    
    # Log transformed distribution
    df[feature + "_log"] = np.log1p(df[feature])
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    sns.histplot(df[feature + "_log"], kde=True)
    plt.title("Log Transformed Distribution")
    
    plt.subplot(1,2,2)
    stats.probplot(df[feature + "_log"], dist="norm", plot=plt)
    plt.title("Log Transformed Probability Plot")
    
    plt.show()
# Apply Transformation
    plot_distribution('policy_annual_premium')