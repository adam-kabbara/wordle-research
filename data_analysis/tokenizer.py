import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import SyllableTokenizer
import ast
from statistics import mean
import numpy as np

# Download required NLTK data
# nltk.download('punkt')

def syllable_tokenizer(word):
    tokenizer = SyllableTokenizer()
    return tokenizer.tokenize(word)

def common_syllables(word1, word2):
    syllables1 = set(syllable_tokenizer(word1))
    syllables2 = set(syllable_tokenizer(word2))
    return len(syllables1.intersection(syllables2))

# Load the data
df = pd.read_csv(r'data_analysis\data\data_1.csv')

# Convert string representation of list to actual list
df['wordle_guesses'] = df['wordle_guesses'].apply(ast.literal_eval)

# Function to calculate common syllables for a game's guesses
def analyze_game_guesses(guesses):
    common_syllables_list = []
    for i in range(1, len(guesses)):
        common = common_syllables(guesses[i-1], guesses[i])
        common_syllables_list.append(common)
    return common_syllables_list

# Apply the analysis to each game
df['guess_syllable_similarities'] = df['wordle_guesses'].apply(analyze_game_guesses)

# Calculate average syllable similarity for each game
df['avg_syllable_similarity'] = df['guess_syllable_similarities'].apply(lambda x: mean(x) if x else 0)

# Create the plot
plt.figure(figsize=(12, 6))
plt.scatter(df['entry_id'], df['avg_syllable_similarity'], alpha=0.5)
plt.title('Average Syllable Similarity Between Consecutive Guesses in Wordle Games')
plt.xlabel('Wordle Game ID')
plt.ylabel('Average Number of Common Syllables')
plt.grid(True, alpha=0.3)

# Add a trend line
z = np.polyfit(df['entry_id'], df['avg_syllable_similarity'], 1)
p = np.poly1d(z)
plt.plot(df['entry_id'], p(df['entry_id']), "r--", alpha=0.8)

# Add some statistics as text
overall_avg = df['avg_syllable_similarity'].mean()
max_avg = df['avg_syllable_similarity'].max()
min_avg = df['avg_syllable_similarity'].min()

stats_text = f'Overall Average: {overall_avg:.2f}\nMax Average: {max_avg:.2f}\nMin Average: {min_avg:.2f}'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=10)

# Save the plot
plt.savefig('wordle_guess_syllable_analysis.png')
plt.close()

# Print some additional statistics
print(f"Total games analyzed: {len(df)}")
print(f"Overall average syllable similarity: {overall_avg:.2f}")

# Find games with highest and lowest average syllable similarity
highest_similarity = df.loc[df['avg_syllable_similarity'] == max_avg].iloc[0]
lowest_similarity = df.loc[df['avg_syllable_similarity'] == min_avg].iloc[0]

print("\nGame with highest average syllable similarity:")
print(f"Wordle {highest_similarity['entry_id']}: {highest_similarity['wordle_guesses']}")
print(f"Answer: {highest_similarity['wordle_answer']}")
print(f"Average similarity: {highest_similarity['avg_syllable_similarity']:.2f}")

print("\nGame with lowest average syllable similarity:")
print(f"Wordle {lowest_similarity['entry_id']}: {lowest_similarity['wordle_guesses']}")
print(f"Answer: {lowest_similarity['wordle_answer']}")
print(f"Average similarity: {lowest_similarity['avg_syllable_similarity']:.2f}")

# Calculate and print overall distribution of syllable similarities
all_similarities = [sim for sims in df['guess_syllable_similarities'] for sim in sims]
similarity_counts = pd.Series(all_similarities).value_counts().sort_index()

print("\nOverall distribution of syllable similarities:")
for similarity, count in similarity_counts.items():
    print(f"{similarity} common syllables: {count} occurrences")