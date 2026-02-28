import seaborn as sns
import matplotlib.pyplot as plt


# Univariate analysis
# Fraud Reported — Countplot
sns.countplot(data=df, x='fraud_reported')
plt.title("Fraud Reported Count")
plt.show()

# Incident Severity — Pie Chart
df['incident_severity'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    figsize=(6,6)
)
plt.title("Damage Visualization")
plt.ylabel("")
plt.show()

# Age Distribution — Histogram
sns.histplot(df['age'], bins=15, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.show()

# Multivariate analysis
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
# Dropping Highly Correlated Features
df = df.drop([
    'months_as_customer',
    'injury_claim',
    'property_claim',
    'vehicle_claim'
], axis=1)