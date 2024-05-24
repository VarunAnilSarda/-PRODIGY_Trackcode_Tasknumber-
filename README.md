import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_df = sns.load_dataset('iris')

print(iris_df.head())

print(iris_df.isnull().sum())

sns.pairplot(iris_df, hue='species', markers=["o", "s", "D"])
plt.title('Pairplot of Iris Dataset')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df, orient="h", palette="Set2")
plt.title('Boxplot of Iris Dataset by Species')
plt.show()

corr_matrix = iris_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Iris Dataset')
plt.show()
