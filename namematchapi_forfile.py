import pandas as pd
import requests

# Load the data
print("Loading data from 'month_data.csv'...")
data = pd.read_csv("nov_data.csv")
print(f"Data loaded successfully. Total rows: {len(data)}")
print(data.head())  # Show the first few rows of the data

def fetch_match_result(name1, name2):
    url = f"http://localhost:3000/match-names/?name1={name1}&name2={name2}"
    try:
        print(f"Fetching match result for name1: {name1}, name2: {name2}...")
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        print(f"Received result for name1: {name1}, name2: {name2}: {result}")
        # Return the necessary columns with specific names
        return {
            'results.first_layer_pass': result.get('First_Layer_Pass'),
            'results.second_layer_score': result.get('Second_Layer_Score'),
            'results.second_layer_pass': result.get('Second_Layer_Pass'),
            'results.third_layer_pass': result.get('Third_Layer_Pass')
        }
    except Exception as e:
        print(f"Error fetching match result for {name1} and {name2}: {e}")
        # Return None for the expected columns in case of an error
        return {
            'results.first_layer_pass': None,
            'results.second_layer_score': None,
            'results.second_layer_pass': None,
            'results.third_layer_pass': None
        }

# Apply the function row-wise
print("Applying fetch_match_result to the DataFrame...")
results = data.apply(lambda row: fetch_match_result(row['name1'], row['name2']), axis=1)
print("API calls completed. Results collected.")

# Convert the list of dictionaries into a DataFrame
print("Converting API results to a DataFrame...")
result_df = pd.DataFrame(results.tolist())
print(f"Results DataFrame created with columns: {result_df.columns.tolist()}")

# Concatenate only the necessary columns with the original DataFrame
print("Concatenating results with the original data...")
final_data = pd.concat([data, result_df], axis=1)
print(f"Final data shape: {final_data.shape}")

# Drop the unwanted columns if they exist
print("Dropping unwanted columns 'result.name1' and 'result.name2' if they exist...")
final_data.drop(columns=['result.name1', 'result.name2'], inplace=True, errors='ignore')
print(f"Final data columns after dropping unwanted columns: {final_data.columns.tolist()}")

# Save the final DataFrame to a CSV file
output_file = "audit_data_result.csv"
print(f"Saving final DataFrame to '{output_file}'...")
final_data.to_csv(output_file, index=False)
print(f"File saved successfully at: {output_file}")
