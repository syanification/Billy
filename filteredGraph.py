import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Load the data
df = pd.read_csv('Training Evaluation/filteredResults.csv')

# Filter out rows where numEpochs > 20
df = df[df['numEpochs'] <= 20]

# Group by modelName and sort each group by numEpochs
groups = df.groupby('modelType')

fig, ax = plt.subplots(figsize=(12, 6))

# Define a colormap for the model names
model_names = df['modelType'].unique()
colors = plt.cm.GnBu(np.linspace(0.4, 1, len(model_names)))
color_map = dict(zip(model_names, colors))

# Plot each model name as a line
for model_name, group in groups:
    sorted_group = group.sort_values('numEpochs')  # Ensure epochs are plotted in order
    ax.plot(
        sorted_group['numEpochs'], 
        sorted_group['testLoss'], 
        linestyle='-', 
        linewidth=1,  # Thinner lines
        alpha=0.7,  # Slightly transparent
        color=color_map[model_name],  # Color based on model name
        label=f'Model = {model_name}'
    )

# Customize the plot
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Test Loss')
ax.set_title('Test Loss vs. Number of Epochs by Model Type')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
ax.grid(True, linestyle='--')

plt.tight_layout()
plt.show()