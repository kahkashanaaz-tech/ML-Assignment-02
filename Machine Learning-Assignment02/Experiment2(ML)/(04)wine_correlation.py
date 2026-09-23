import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
correlation = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Wine Dataset")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
corr_matrix = correlation.copy()
for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = 0
feature1, feature2 = corr_matrix.stack().idxmax()
strongest = corr_matrix.loc[feature1, feature2]

print("\nStrongest Positive Correlation:")
print(feature1, "and", feature2)
print("Correlation value:", round(strongest, 2))
