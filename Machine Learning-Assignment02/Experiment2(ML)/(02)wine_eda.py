import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# 1. Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

# 2. Display first five rows
print("--- First Five Rows ---")
print(df.head())

# 3. Dataset information
print("\n--- Dataset Information ---")
print(df.info())

# 4. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 5. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 6. Check duplicate values
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 7. Target class distribution
print("\n--- Target Class Distribution ---")
print(df["target"].value_counts())

# 8. Correlation matrix
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# 9. Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix of Wine Dataset")
plt.show()

# 10. Histogram
plt.figure(figsize=(7, 5))
sns.histplot(df["alcohol"], kde=True)
plt.title("Distribution of Alcohol")
plt.xlabel("Alcohol")
plt.show()

# 11. Scatter plot
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="alcohol",
    y="malic_acid",
    hue="target"
)
plt.title("Alcohol vs Malic Acid")
plt.show()