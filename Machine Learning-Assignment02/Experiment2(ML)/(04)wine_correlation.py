import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Calculate correlation matrix
correlation = df.corr()

# Generate correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Wine Dataset")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Find strongest positive correlation
corr_matrix = correlation.copy()

# Remove diagonal values
for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = 0

# Find pair with highest positive correlation
feature1, feature2 = corr_matrix.stack().idxmax()
strongest = corr_matrix.loc[feature1, feature2]

print("\nStrongest Positive Correlation:")
print(feature1, "and", feature2)
print("Correlation value:", round(strongest, 2))