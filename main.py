import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = sns.load_dataset("iris")

# Display the first few rows of the dataset
print(iris_data.head())

# Summary statistics of the dataset
print("\nSummary Statistics:")
print(iris_data.describe())

# Pair plot to visualize relationships between numerical features
sns.pairplot(iris_data, hue="species")
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Box plots to visualize the distribution of each feature for different species
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, feature in enumerate(iris_data.columns[:-1]):
    sns.boxplot(x="species", y=feature, data=iris_data, ax=axes[i // 2, i % 2])

fig.suptitle("Box Plots of Iris Dataset Features by Species", y=1.02)
plt.show()

# Correlation heatmap to visualize the correlation between numerical features
correlation_matrix = iris_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()

# Violin plots to visualize the distribution of each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, feature in enumerate(iris_data.columns[:-1]):
    sns.violinplot(x="species", y=feature, data=iris_data, ax=axes[i // 2, i % 2])

fig.suptitle("Violin Plots of Iris Dataset Features by Species", y=1.02)
plt.show()

