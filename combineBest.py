'''
This file combines all 0.01 learning rate values from the training results to see which model performs the best
'''

import pandas as pd

# List of input CSV files
input_files = [
    'Training Evaluation/trainingResultsDNN.csv',
    'Training Evaluation/trainingResultsDNNDropout.csv',
    'Training Evaluation/trainingResultsLinear.csv'
]

# Output CSV file
output_file = 'Training Evaluation/filteredResults.csv'

# List to hold filtered dataframes
filtered_dfs = []

for file in input_files:
    # Read CSV
    df = pd.read_csv(file)
    
    # Filter rows with learningRate = 0.01
    filtered = df[df['learningRate'] == 0.01]
    
    if not filtered.empty:
        # Add source file column
        if file == input_files[0]:
            modelType = "DNN"
        elif file == input_files[1]:
            modelType = "DNN + Dropout"
        else:
            modelType = "Linear"

        filtered = filtered.assign(modelType = modelType)
        
        # Remove learningRate column
        filtered = filtered.drop(columns=['learningRate'])
        
        filtered_dfs.append(filtered)

# Combine all filtered data
combined_df = pd.concat(filtered_dfs, ignore_index=True)

# Reorder columns (optional)
combined_df = combined_df[['modelType', 'numEpochs', 'testLoss', 'testMae']]

# Save to new CSV
combined_df.to_csv(output_file, index=False)

print(f"Combined filtered results saved to {output_file}!")