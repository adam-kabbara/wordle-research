import pandas as pd
import numpy as np
import gensim.downloader as api
import seaborn as sns
from tqdm import tqdm


glove_model = api.load('glove-wiki-gigaword-300')
w2v_model = api.load('word2vec-google-news-300')
word_list = '/Users/jessica_1/Workspace/Wordle_project/wordle-nyt-answers-alphabetical.txt'

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to compute distance between two words using GloVe
def glove_distance(word1, word2, model):
    if word1 in model and word2 in model:
        vec1 = model[word1]
        vec2 = model[word2]
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(vec1, vec2)
        distance = 1 - similarity  # Cosine distance

        return similarity, distance  # Return if both words are found
    else:
        # Return None for similarity and distance if words are not found
        return None, None

if __name__ == "__main__":
    # Read words from the word list file
    word_list = '/Users/jessica_1/Workspace/Wordle_project/wordle-nyt-answers-alphabetical.txt'
    
    with open(word_list, 'r') as f:
        words = f.read().splitlines()

    test_words = words[:4]

    df_results = pd.DataFrame(columns=['Word1', 'Word2', 'Similarity', 'Distance'])

    total_comparisons = (len(test_words) * (len(test_words) - 1)) // 2  # Total number of comparisons
    # progress_bar = tqdm(total=total_comparisons, desc="Comparing words", unit="comparison", leave=True, mininterval=0.1, ncols=100)

    for i in tqdm(range(len(test_words)), total=len(test_words), desc="Comparing words", unit="word"):
        for j in range(i + 1, len(test_words)):  
            word1 = test_words[i]
            word2 = test_words[j]
            similarity, distance = glove_distance(word1, word2, glove_model)

            if similarity is not None: 
                new_row = pd.DataFrame({
                    'Word1': [word1],
                    'Word2': [word2],
                    'Similarity': [similarity],
                    'Distance': [distance]
                })
                # Use pd.concat() to append the new row to df_results
                df_results = pd.concat([df_results, new_row], ignore_index=True)
            else:
                print(f"One or both words ('{word1}', '{word2}') not found in the GloVe model.")

    #         progress_bar.update(1)
    #         progress_bar.refresh()

    # progress_bar.close()

    glove_avg_similarity = df_results['Similarity'].mean()
    glove_std_similarity = df_results['Similarity'].std()

    glove_avg_distance = df_results['Distance'].mean()
    glove_std_distance = df_results['Distance'].std()

    print(f"total possible combination of words: {total_comparisons}")
    print(f"number of available pairs: {df_results.shape[0]}")
    print(f"glove average similarity: {glove_avg_similarity}")
    print(f"glove similarity std dev: {glove_std_similarity}")

    print(f"glove average distance: {glove_avg_distance}")
    print(f"glove distance std dev: {glove_std_distance}")

    # Display the DataFrame to verify
    print(df_results.head())

# def word2vec_distance(word1, word2, model):
#     if word1 in model and word2 in model:
#         vec1 = model[word1]
#         vec2 = model[word2]
        
#         # Calculate Cosine Similarity
#         similarity = cosine_similarity(vec1, vec2)
#         distance = 1 - similarity  # Cosine distance
        
#         return similarity, distance
#     else:
#         return None, None 