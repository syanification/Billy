import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Load the data
df = pd.read_csv('trainingResultsLinear.csv')

df = df[df['numEpochs'] >= 20]

# Group by learningRate and sort each group by numEpochs
groups = df.groupby('learningRate')

fig, ax = plt.subplots(figsize=(12, 6))

# Original GnBu colormap
cmap_orig = plt.get_cmap('GnBu')

# Slice the colormap between 0.3 and 1.0 to skip the first 30%
colors_shifted = cmap_orig(np.linspace(0.5, 1, 256))
cmap_shifted = mcolors.ListedColormap(colors_shifted)

norm = plt.Normalize(vmin=df['learningRate'].min(), vmax=df['learningRate'].max())
sm = plt.cm.ScalarMappable(cmap=cmap_shifted, norm=norm)
sm.set_array([])

# Plot each learning rate as a line
for lr, group in groups:
    sorted_group = group.sort_values('numEpochs')  # Ensure epochs are plotted in order
    ax.plot(
        sorted_group['numEpochs'], 
        sorted_group['testLoss'], 
        linestyle='-', 
        linewidth=1,  # Thinner lines
        alpha=0.7,  # Slightly transparent
        color=cmap_shifted(norm(lr)),  # Gradient color
        label=f'LR = {lr}'
    )

# Customize the plot
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Test Loss')
ax.set_title('Test Loss vs. Number of Epochs by Learning Rate (DNN)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
ax.grid(True, linestyle='--')

# Only one colorbar call, tied to the figure and ax
# cbar = fig.colorbar(sm, ax=ax)
# cbar.set_label('Learning Rate')

plt.tight_layout()
plt.show()