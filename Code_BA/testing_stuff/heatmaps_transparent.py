import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate some example data
np.random.seed(0)
data = np.random.rand(10, 10)  # mean values
uncertainty = np.random.rand(10, 10) * 0.2  # uncertainty values

# Normalize uncertainty for alpha values
norm_uncertainty = Normalize()(uncertainty)

# Plot heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(data, cmap="viridis", annot=True, cbar=False)

# Overlay alpha transparency
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        heatmap.add_patch(plt.Rectangle((j, i), 1, 1, color='white', alpha=1-norm_uncertainty[i, j]))

plt.title('Heatmap with Uncertainty Represented by Transparency')
plt.show()
