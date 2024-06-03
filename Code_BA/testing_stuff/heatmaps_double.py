import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
data = np.random.rand(10, 10)  # mean values
uncertainty = np.random.rand(10, 10) * 0.2  # uncertainty values

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Primary heatmap for the mean values
sns.heatmap(data, ax=ax1, cmap="viridis", annot=True)
ax1.set_title('Mean Values')

# Heatmap for the uncertainty
sns.heatmap(uncertainty, ax=ax2, cmap="coolwarm", annot=True)
ax2.set_title('Uncertainty')

plt.show()