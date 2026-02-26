import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def boxplot(df, column):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def distribution_plot(df, column):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Distribution")

    plt.subplot(1,2,2)
    stats.probplot(df[column], dist="norm", plot=plt)

    plt.show()