'''
This file runs through each line of common2.csv and finds the lowest loss players from the model
'''
import pandas as pd
from google.cloud import aiplatform
from google.oauth2 import service_account
import json

# Load the JSON credentials file
with open('Auth/owner-auth.json') as f:
    credentials_dict = json.load(f)

# Create credentials object
credentials = service_account.Credentials.from_service_account_info(credentials_dict)

PROJECT_NUMBER = 448750250460
ENDPOINT_ID = 7943808782960689152

aiplatform.init(
    project=PROJECT_NUMBER,
    location='northamerica-northeast2',
    credentials=credentials
)

endpoint = aiplatform.Endpoint(
    endpoint_name=f"projects/{PROJECT_NUMBER}/locations/northamerica-northeast2/endpoints/{ENDPOINT_ID}"
)

# Load the data
df = pd.read_csv('Data/common2.csv')

# List to store results
results = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():

    if row['PA.y'] < 500: continue

    # Prepare the input for prediction
    input_data = [[row['BA.x'], row['OBP.x'], row['SLG.x']]]
    
    # Make prediction
    prediction = endpoint.predict(instances=input_data).predictions[0]
    
    # Calculate the individual differences
    diff_ba = abs(row['BA.x'] - prediction[0])
    diff_obp = abs(row['OBP.x'] - prediction[1])
    diff_slg = abs(row['SLG.x'] - prediction[2])

    # Calculate the total difference
    diff = diff_ba + diff_obp + diff_slg

    # Print the individual differences and the total difference
    print(f"Name: {row['Name']}, BA.x: {row['BA.x']}, Predicted_BA.y: {prediction[0]}, Difference_BA: {diff_ba}")
    print(f"OBP.x: {row['OBP.x']}, Predicted_OBP.y: {prediction[1]}, Difference_OBP: {diff_obp}")
    print(f"SLG.x: {row['SLG.x']}, Predicted_SLG.y: {prediction[2]}, Difference_SLG: {diff_slg}")
    print(f"Total Difference: {diff}")

    # Store the result
    results.append({
        'Name': row['Name'],
        'BA.x': row['BA.x'],
        'Predicted_BA.y': prediction[0],
        'BA.y': row['BA.y'],
        'OBP.x': row['OBP.x'],
        'Predicted_OBP.y': prediction[1],
        'OBP.y': row['OBP.y'],
        'SLG.x': row['SLG.x'],
        'Predicted_SLG.y': prediction[2],
        'SLG.y': row['SLG.y'],
        'Difference': diff
    })

# Sort the results by the difference
sorted_results = sorted(results, key=lambda x: x['Difference'])

# Select the top 100 names with the lowest difference
top = sorted_results[:100]

# Print the top 100 results
for result in top:
    print(result)

# Save the top results to a CSV file
top_df = pd.DataFrame(top)
top_df.to_csv('Data/bestfit.csv', index=False)