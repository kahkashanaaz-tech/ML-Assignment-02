import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
print("--- First Five Rows ---")
print(df.head())
print("\n--- Dataset Information ---")
print(df.info())
print("\n--- Statistical Summary ---")
print(df.describe())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())
print("\n--- Target Class Distribution ---")
print(df["target"].value_counts())
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix of Wine Dataset")
plt.show()
plt.figure(figsize=(7, 5))
sns.histplot(df["alcohol"], kde=True)
plt.title("Distribution of Alcohol")
plt.xlabel("Alcohol")
plt.show()
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="alcohol",
    y="malic_acid",
    hue="target"
)
plt.title("Alcohol vs Malic Acid")
plt.show()
