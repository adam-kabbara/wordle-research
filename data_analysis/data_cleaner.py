import pandas as pd
import re

df = pd.read_csv(r"data_analysis\data\processed_wordle_data.csv")
answers_df = pd.read_csv(r"data_analysis\data\answers.csv")

new_df = pd.DataFrame()

# Copy the specified columns from the original DataFrame
new_df["wordle_guesses"] = df["wordle_guesses"] if "wordle_guesses" in df else "unknown"
new_df["num_guesses"] = df["num_guesses"] if "num_guesses" in df else "unknown"
new_df["wordle_answer"] = df["wordle_answer"] if "wordle_answer" in df else "unknown"
new_df["author"] = df["author"] if "author" in df else "unknown"

# Define regex patterns
title_pattern = r'^(Wordle)\s+(\d+)'
asterisk_pattern = lambda wordle_id: rf'{wordle_id}.{{0,15}}\*'


# Function to extract Wordle title, ID, and hard mode status
def extract_wordle_info(body_text):
    # Default values
    wordle_title = "unknown"
    wordle_id = "unknown"
    hard_mode = False
    
    # Extract title and wordle_id
    match = re.search(title_pattern, body_text, re.MULTILINE)
    if match:
        wordle_title = match.group(1)
        wordle_id = match.group(2)
    
    # Check for hard mode (asterisk pattern)
    if wordle_id != "unknown":
        if re.search(asterisk_pattern(wordle_id), body_text):
            hard_mode = True
    
    return wordle_title, wordle_id, hard_mode

def get_wordle_answer(row):
    if row["wordle_answer"] != "unknown":
        return row["wordle_answer"]
    elif row["wordle_guesses"] != "unknown":
        # Search in answers_df for the row with the same id
        match = answers_df[answers_df["id"] == row["wordle_id"]]
        if not match.empty:
            return match.iloc[0]["word"].lower()
        else:
            return "unknown"
    else:
        return "unknown"
    


new_df = pd.DataFrame()

# Copy the specified columns from the original DataFrame
new_df["entry_id"] = range(1, len(df) + 1)
new_df["author"] = df["author"] if "author" in df else "unknown"
new_df["wordle_guesses"] = df["wordle_guesses"] if "wordle_guesses" in df else "unknown"
new_df["num_guesses"] = df["num_guesses"] if "num_guesses" in df else "unknown"
new_df["wordle_answer"] = df["wordle_answer"] if "wordle_answer" in df else "unknown"

# Define regex patterns
title_pattern = r'^(Wordle)\s+(\d+)'
asterisk_pattern = lambda wordle_id: rf'{wordle_id}.{{0,15}}\*'

# Apply the function to each row of the original DataFrame and convert the result to a list of tuples
extracted_info = df["body"].apply(extract_wordle_info)

# Unpack the list of tuples into separate columns
new_df["wordle_title"], new_df["wordle_id"], new_df["hard_mode"] = zip(*extracted_info)

# Convert hard_mode to boolean strings
new_df["hard_mode"] = new_df["hard_mode"].apply(lambda x: "true" if x else "false")

# Get answers from answers_df
new_df["wordle_answer"] = new_df.apply(get_wordle_answer, axis=1)

# Save the new DataFrame to CSV (optional)
new_df.to_csv('new_wordle_data.csv', index=False)

# Display the new DataFrame
print(new_df)