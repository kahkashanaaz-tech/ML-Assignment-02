import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

# Load Wine dataset
wine = load_wine()

# Create DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Plot boxplots for all numerical attributes
plt.figure(figsize=(15, 8))

sns.boxplot(data=df)

plt.title("Boxplots of Wine Dataset Numerical Attributes")
plt.xticks(rotation=90)
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.tight_layout()

plt.show()