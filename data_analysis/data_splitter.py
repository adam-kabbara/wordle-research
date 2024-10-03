# split the data into 50 batches and save them into new csv's

import pandas as pd
import os

# Split the data into 50 batches

def split_data(data, store_dir="data_analysis/data/data_batches", num_batches=50):
    total_rows = len(data)
    batch_size = total_rows // num_batches
    for i in range(50):
        start = i * batch_size
        # Ensure the last batch captures any remaining rows
        end = start + batch_size if i < 49 else total_rows
        data_batch = data.iloc[start:end]
        data_batch.to_csv(os.path.join(store_dir, f'data_batch_{i}.csv'), index=False, header=True)


def check_line_count(directory, initial_csv):
    # Load the initial data and get the number of rows (excluding header)
    initial_data = pd.read_csv(initial_csv)
    initial_row_count = len(initial_data)

    # Initialize a counter for total rows from the CSVs in the directory
    total_row_count = 0

    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            # Read each CSV file and count the number of rows (excluding header)
            batch_data = pd.read_csv(file_path)
            total_row_count += len(batch_data)

    # Print the results
    print(f'Initial row count (excluding header): {initial_row_count}')
    print(f'Total row count from CSVs in directory: {total_row_count}')

    # Check if the counts match
    if initial_row_count == total_row_count:
        print("All data has been successfully split and no rows have been lost.")
    else:
        print("Warning: The counts do not match. Data may be missing.")

# Load the data
init_file = 'data_analysis/data/data_1.csv'
store_dir = 'data_analysis/data/data_batches'
data = pd.read_csv(init_file)
split_data(data, store_dir=store_dir, num_batches=50)

# Example usage
check_line_count(store_dir, init_file)