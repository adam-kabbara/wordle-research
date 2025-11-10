"""
TODO:
- for rhyming avg you can just like avg u need to take avg of count - cause rhyming is tru or false its not a value
"""




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List, Tuple, Dict, Union
from ast import literal_eval
from nltk.tokenize import SyllableTokenizer
import numpy as np
from numpy.linalg import norm
import tqdm
import gensim.downloader as api
from numpy.linalg import norm
import numpy as np
import pickle
import os
from scipy import stats
from scipy.stats import ttest_rel
import nltk
from nltk.corpus import wordnet as wn
from transformers import GPT2Tokenizer
import pronouncing
from collections import defaultdict


class WordleAnalyzer:
    print("Loading models...")
    syllable_tokenizer = SyllableTokenizer()
    glove_distance_model = api.load('glove-wiki-gigaword-300')
    word2vec_model = api.load('word2vec-google-news-300')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    nltk.download('wordnet')
    nltk.download('punkt')
    sns.set_palette("bright")
    MAX_DIST = 2

    def __init__(self, csv_path: str, load_pickle: bool = True, pickle_name: str = "pickled_data.pkl", main_dir: str = "data_analysis/generated_data", avg_func: str = "mean", avg_dec_place:int=1, specific_dec_places: Dict[str, int] = None):
        """Initialize the WordleAnalyzer with a CSV file path."""
        print("Loading data...")
        self.df = pd.read_csv(csv_path)
        print("Preprocessing data...")
        self._preprocess_data()
        self.metrics = {}
        self.main_dir = main_dir
        os.makedirs(main_dir, exist_ok=True)
        self.pickle_path = os.path.join(main_dir, pickle_name)
        self.avg_func = avg_func

        if specific_dec_places is not None: # Use specific decimal places for certain metrics
            self.dec_places = defaultdict(lambda: avg_dec_place, specific_dec_places)
        else: # Use the same decimal places for all metrics
            self.dec_places = defaultdict(lambda: avg_dec_place)
        
        if load_pickle and os.path.exists(self.pickle_path):
            self.load_pickled_metrics()

    def save_plot(self, plot_func: Callable[[plt.Figure], None], filename: str) -> None:
        """Save a plot as PDF in the main directory."""
        fig = plt.figure(figsize=(10, 10))
        plot_func(fig)
        pdf_path = os.path.join(self.main_dir, f"{filename}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
    
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Convert string representations of lists to actual lists
        self.df['wordle_guesses'] = self.df['wordle_guesses'].apply(literal_eval)
        self.df['optimal'] = self.df['optimal'].apply(literal_eval)

        # convert all string instances to lowercase in all of the dataframes
        self.df['wordle_guesses'] = self.df['wordle_guesses'].apply(lambda x: [i.lower() for i in x])
        self.df['optimal'] = self.df['optimal'].apply(lambda x: [[(word.lower(), count) for word, count in sublist] for sublist in x])
        self.df['wordle_answer'] = self.df['wordle_answer'].apply(lambda x: x.lower())

    def dump_pickle_metrics(self):
        """Save metrics to pickle file."""
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.metrics, f)
            print(f"Metrics saved to {self.pickle_path}")

    def load_pickled_metrics(self):
        """Load metrics from pickle file."""
        try:
            with open(self.pickle_path, 'rb') as f:
                self.metrics = pickle.load(f)
                print(f"Metrics loaded from pickel file: {self.pickle_path}")
        except (FileNotFoundError, EOFError) as e:
            print(f"Error loading pickle file: {e}")
            self.metrics = {}
    
    @staticmethod
    def levenshtein(source: str, target: str) -> int:
        """Calculate Levenshtein distance between two words."""
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)
        if source[0] == target[0]:
            return WordleAnalyzer.levenshtein(source[1:], target[1:])
        direct_edit = WordleAnalyzer.levenshtein(source[1:], target[1:])
        insert = WordleAnalyzer.levenshtein(source, target[1:])
        delete = WordleAnalyzer.levenshtein(source[1:], target)
        return 1 + min(delete, min(direct_edit, insert))
    
    @staticmethod
    def avg_levenshtein_within(guess_list: List[str], start_idx: int, func: str = "mean", dec_place: int=1) -> Union[float, str]:
        """Calculate average Levenshtein distance within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no distance")
        
        values = np.array([WordleAnalyzer.levenshtein(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place)
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)
        else:
            raise("Invalid function")
    
    @staticmethod
    def common_syllables(word1: str, word2: str) -> int:
        """Calculate number of common syllables between two words."""
        syllables1 = set(WordleAnalyzer.syllable_tokenizer.tokenize(word1))
        syllables2 = set(WordleAnalyzer.syllable_tokenizer.tokenize(word2))
        return len(syllables1.intersection(syllables2))
    
    @staticmethod
    def avg_common_syllables_within(guess_list: List[str], start_idx: int, func: str = "mean", dec_place: int=1) -> Union[float, str]:
        """Calculate average number of common syllables within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no common syllables")
        
        values = np.array([WordleAnalyzer.common_syllables(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place)
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)
        else:
            raise("Invalid function")
    
    @staticmethod
    def common_gpt_tokens(word1: str, word2: str) -> int:
        """Calculate number of common GPT tokens between two words."""
        tokens1 = set(word1.split())
        tokens2 = set(word2.split())
        return len(tokens1.intersection(tokens2))
    
    @staticmethod
    def avg_common_gpt_tokens_within(guess_list: List[str], start_idx: int, func:str = "mean", dec_place: int=1) -> Union[float, str]:
        """Calculate average number of common GPT tokens within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no common GPT tokens")
        
        values = np.array([WordleAnalyzer.common_gpt_tokens(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place)
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)
        else:
            raise("Invalid function")
    
    @staticmethod
    def shared_chars(word1: str, word2: str) -> int:
        """Calculate number of shared characters between two words."""
        return len(set(word1).intersection(set(word2)))
    
    @staticmethod
    def avg_shared_chars_within(guess_list: List[str], start_idx: int, func:str = "mean", dec_place: int=1) -> Union[float, str]:
        """Calculate average number of shared characters within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no shared characters")
        
        values = np.array([WordleAnalyzer.shared_chars(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place)
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)
        else:
            raise("Invalid function")
    
    @staticmethod
    def is_rhyme_0(text1, text2):
        return int(text2 in pronouncing.rhymes(text1) or text1 in pronouncing.rhymes(text2))
    
    @staticmethod
    def get_phonetic_ending(word):
        phones = pronouncing.phones_for_word(word)
        
        if not phones:
            return None
        
        phonetic_transcription = phones[0].split()
        
        # Find the index of the last stressed vowel (vowels with 1 or 2 indicate stress)
        stressed_vowel_index = None
        for i, phoneme in enumerate(phonetic_transcription):
            if phoneme[-1] in "012":  # Stressed vowels end in 1 or 2
                stressed_vowel_index = i
        
        if stressed_vowel_index is not None:
            return phonetic_transcription[stressed_vowel_index:]
        else:
            return None  # Return None if no stressed vowel is found
    
    @staticmethod
    def is_rhyme_1(text1, text2):
        ending1 = WordleAnalyzer.get_phonetic_ending(text1)
        ending2 = WordleAnalyzer.get_phonetic_ending(text2)
        
        if ending1 and ending2:
            return int(ending1 == ending2)  # Check if the phonetic endings match
        else:
            return 0
        
    
    @staticmethod
    def avg_rhyme_count(guess_list: List[str], start_idx: int, func: str = "mean", rhyme_func = 0, dec_place=1) -> Union[float, str]:
        """Calculate average number of rhymes within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no rhyme")
        
        if rhyme_func == 0:
            values = np.array([WordleAnalyzer.is_rhyme_0(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        elif rhyme_func == 1:
            values = np.array([WordleAnalyzer.is_rhyme_1(guess_list[i], guess_list[i+1]) for i in range(start_idx)])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place)
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)
        else:
            raise("Invalid function")
    
        
    @staticmethod
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    # Function to compute distance between two words using GloVe
    @staticmethod
    def glove_distance(word1: str, word2: str, model=None, dec_place=1) -> float:
        """Compute distance between two words using GloVe embeddings."""
        if model is None:
            model = WordleAnalyzer.glove_distance_model
        if word1 in model and word2 in model:
            vec1 = model[word1]
            vec2 = model[word2]
            similarity = WordleAnalyzer.cosine_similarity(vec1, vec2)
            return round(1 - similarity, dec_place)
        return None #WordleAnalyzer.MAX_DIST
        # todo we assume that if the word is not in the model than the two words have max distance which is 2
    @staticmethod
    def avg_glove_distance_within(guess_list: List[str], start_idx: int, model, func: str = "mean", dec_place:int=1) -> Union[float, str]:
        """Calculate average GloVe distance within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no distance")
        
        values = np.array([WordleAnalyzer.glove_distance(guess_list[i], guess_list[i+1], model, dec_place=100) for i in range(start_idx) if WordleAnalyzer.glove_distance(guess_list[i], guess_list[i+1], model, dec_place=100) is not None])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place) 
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)   

    @staticmethod
    def calculate_semantic_similarities(word1: str, word2: str, dec_place=1) -> float:
        synsets1 = wn.synsets(word1) # todo this and the avg for it 
        synsets2 = wn.synsets(word2)
        
        # Ensure words have synsets (not all words exist in WordNet)
        if not synsets1 or not synsets2:
            return None
        
        # Initialize maximum similarity
        max_path_sim = 0
        max_wup_sim = 0
        max_lch_sim = 0
        
        # Iterate through all synset pairs and calculate the similarity
        for synset1 in synsets1:
            for synset2 in synsets2:
                path_sim = synset1.path_similarity(synset2)
                wup_sim = synset1.wup_similarity(synset2)
                try:
                    lch_sim = synset1.lch_similarity(synset2)
                except:
                    lch_sim = None
                
                # Update maximum similarity found
                if path_sim is not None and path_sim > max_path_sim:
                    max_path_sim = round(path_sim, dec_place)
                if wup_sim is not None and wup_sim > max_wup_sim:
                    max_wup_sim = round(wup_sim, dec_place)
                if lch_sim is not None and lch_sim > max_lch_sim:
                    max_lch_sim = round(lch_sim, dec_place)
        
        return {
            'Path Similarity': max_path_sim,
            'Wu-Palmer Similarity': max_wup_sim,
            'Leacock-Chodorow Similarity': max_lch_sim
        }
    
    @staticmethod
    def word2vec_distance(word1, word2, model, dec_place=1):
        if word1 in model and word2 in model:
            vec1 = model[word1]
            vec2 = model[word2]
            
            # Calculate Cosine Similarity
            similarity = WordleAnalyzer.cosine_similarity(vec1, vec2)
            distance = 1 - similarity  # Cosine distance
            
            return round(distance, dec_place)
        return None #WordleAnalyzer.MAX_DIST
    # todo we assume that if the word is not in the model than the two words have max distance which is 2 
    @staticmethod
    def avg_word2vec_distance_within(guess_list: List[str], start_idx: int, model, func:str="mean", dec_place:int=1) -> Union[float, str]:
        """Calculate average Word2Vec distance within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            raise("no word2vec distance")
        
        values = np.array([WordleAnalyzer.word2vec_distance(guess_list[i], guess_list[i+1], model, dec_place=100) for i in range(start_idx) if WordleAnalyzer.word2vec_distance(guess_list[i], guess_list[i+1], model, dec_place=100) is not None])
        if func == "mean":
            return round(values.mean(), dec_place)
        elif func == "median":
            return round(np.median(values), dec_place) 
        elif func == "mode":
            return round(stats.mode(values).mode, dec_place)   
    
    def get_optimal_guesses(self, game_index: int) -> List[str]:
        """Get optimal guesses for game at specified index."""
        row = self.df.iloc[game_index]
        actual_guesses = row['wordle_guesses']
        games = []
        for i in range(len(actual_guesses)-1):
            games.append(actual_guesses[:i+1] + [row['optimal'][i][1][0]])
        return games

    def get_comparison_metrics(self, game_index: int) -> Dict[str, List[float]]:
        """Get comparison metrics for optimal vs actual game at specified index."""        
        row = self.df.iloc[game_index]
        actual_guesses = row['wordle_guesses']
        optimal_sequence = self.get_optimal_guesses(game_index)
        metrics = {
            # levenshtein and acg_levenshtein_within
            'actual_levenshtein': [],
            'actual_avg_levenshtein': [],
            'optimal_levenshtein': [],
            'optimal_avg_levenshtein': [],

            # common_syllables and avg_common_syllables_within
            'actual_syllables': [],
            'actual_avg_syllables': [],
            'optimal_syllables': [],
            'optimal_avg_syllables': [],

            # shared_chars and avg_shared_chars_within
            'actual_shared_chars': [], 
            'actual_avg_shared_chars': [],
            'optimal_shared_chars': [],
            'optimal_avg_shared_chars': [],

            # glove_distance and avg_glove_distance_within
            'actual_glove_distance': [], 
            'actual_avg_glove_distance': [],
            'optimal_glove_distance': [],
            'optimal_avg_glove_distance': [],

            # word2vec_distance and avg_word2vec_distance_within
            'actual_word2vec_distance': [], 
            'actual_avg_word2vec_distance': [],
            'optimal_word2vec_distance': [],
            'optimal_avg_word2vec_distance': [],

            # is_rhyme_0 and avg_rhyme_count
            'actual_rhyme0_count': [], 
            'actual_avg_rhyme0_count': [],
            'optimal_rhyme0_count': [],
            'optimal_avg_rhyme0_count': [],

            # is_rhyme_1 and avg_rhyme_count
            'actual_rhyme1_count': [], 
            'actual_avg_rhyme1_count': [],
            'optimal_rhyme1_count': [],
            'optimal_avg_rhyme1_count': [],

            # common_gpt_tokens and avg_common_gpt_tokens_within
            'actual_gpt_tokens': [], 
            'actual_avg_gpt_tokens': [],
            'optimal_gpt_tokens': [],
            'optimal_avg_gpt_tokens': []

        }
        
        # Calculate metrics for actual and optimal guesses between i and i+1
        for i in range(len(actual_guesses) - 1): # = len(optimal_sequence)
            cur_game = optimal_sequence[i] # cur optimal game
            l = len(cur_game)-1
            metrics['actual_levenshtein'].append(self.levenshtein(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_levenshtein'].append(self.levenshtein(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_levenshtein'].append(self.avg_levenshtein_within(actual_guesses, +1, self.avg_func, self.dec_places["actual_avg_levenshtein"]))
            metrics['optimal_avg_levenshtein'].append(self.avg_levenshtein_within(cur_game, l, self.avg_func, self.dec_places["optimal_avg_levenshtein"]))
    
            metrics['actual_syllables'].append(self.common_syllables(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_syllables'].append(self.common_syllables(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_syllables'].append(self.avg_common_syllables_within(actual_guesses, i+1, self.avg_func, self.dec_places["actual_avg_syllables"]))
            metrics['optimal_avg_syllables'].append(self.avg_common_syllables_within(cur_game, l, self.avg_func, self.dec_places["optimal_avg_syllables"]))

            metrics['actual_shared_chars'].append(self.shared_chars(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_shared_chars'].append(self.shared_chars(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_shared_chars'].append(self.avg_shared_chars_within(actual_guesses, i+1, self.avg_func, self.dec_places["actual_avg_shared_chars"]))
            metrics['optimal_avg_shared_chars'].append(self.avg_shared_chars_within(cur_game, l, self.avg_func, self.dec_places["optimal_avg_shared_chars"]))

            temp_actual = self.glove_distance(actual_guesses[i], actual_guesses[i+1], WordleAnalyzer.glove_distance_model)
            temp_guess = self.glove_distance(cur_game[-1], cur_game[-2], WordleAnalyzer.glove_distance_model)
            if temp_actual is not None and temp_guess is not None:
                metrics['actual_glove_distance'].append(temp_actual)
                metrics['optimal_glove_distance'].append(temp_guess)
            metrics['actual_avg_glove_distance'].append(self.avg_glove_distance_within(actual_guesses, i+1, WordleAnalyzer.glove_distance_model, self.avg_func, self.dec_places["actual_avg_glove_distance"]))
            metrics['optimal_avg_glove_distance'].append(self.avg_glove_distance_within(cur_game, l, WordleAnalyzer.glove_distance_model, self.avg_func, self.dec_places["optimal_avg_glove_distance"]))

            temp_actual = self.word2vec_distance(actual_guesses[i], actual_guesses[i+1], WordleAnalyzer.word2vec_model)
            temp_guess = self.word2vec_distance(cur_game[-1], cur_game[-2], WordleAnalyzer.word2vec_model)
            if temp_actual is not None and temp_guess is not None:
                metrics['actual_word2vec_distance'].append(temp_actual)
                metrics['optimal_word2vec_distance'].append(temp_guess)
            metrics['actual_avg_word2vec_distance'].append(self.avg_word2vec_distance_within(actual_guesses, i+1, WordleAnalyzer.word2vec_model, self.avg_func, self.dec_places["actual_avg_word2vec_distance"]))
            metrics['optimal_avg_word2vec_distance'].append(self.avg_word2vec_distance_within(cur_game, l, WordleAnalyzer.word2vec_model, self.avg_func, self.dec_places["optimal_avg_word2vec_distance"]))

            metrics['actual_rhyme0_count'].append(self.is_rhyme_0(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_rhyme0_count'].append(self.is_rhyme_0(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_rhyme0_count'].append(self.avg_rhyme_count(actual_guesses, i+1, self.avg_func, 0, self.dec_places["actual_avg_rhyme0_count"]))
            metrics['optimal_avg_rhyme0_count'].append(self.avg_rhyme_count(cur_game, l, self.avg_func, 0, self.dec_places["optimal_avg_rhyme0_count"]))

            metrics['actual_rhyme1_count'].append(self.is_rhyme_1(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_rhyme1_count'].append(self.is_rhyme_1(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_rhyme1_count'].append(self.avg_rhyme_count(actual_guesses, i+1, self.avg_func, 1, self.dec_places["actual_avg_rhyme1_count"]))
            metrics['optimal_avg_rhyme1_count'].append(self.avg_rhyme_count(cur_game, l, self.avg_func, 1, self.dec_places["optimal_avg_rhyme1_count"]))

            metrics['actual_gpt_tokens'].append(self.common_gpt_tokens(actual_guesses[i], actual_guesses[i+1]))
            metrics['optimal_gpt_tokens'].append(self.common_gpt_tokens(cur_game[-1], cur_game[-2]))
            metrics['actual_avg_gpt_tokens'].append(self.avg_common_gpt_tokens_within(actual_guesses, i+1, self.avg_func, self.dec_places["actual_avg_gpt_tokens"]))
            metrics['optimal_avg_gpt_tokens'].append(self.avg_common_gpt_tokens_within(cur_game, l, self.avg_func, self.dec_places["optimal_avg_gpt_tokens"]))            
            
        self.metrics[game_index] = metrics
        return metrics
    
    def plot_comparison_scatter(self, metric_type: str, n_games: int = None, sizes: Tuple[int, int] = (80, 800), save_pdf: bool = False, save_data: bool = False):
        """Create scatter plot with regression line comparing optimal vs actual games."""
        print(f"Plotting comparison scatter for {metric_type}...")  

        if n_games is None:
            n_games = len(self.df)

        actual_data = []
        optimal_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)

        # Collect data
        for i in tqdm.tqdm(sampled_indices):
            if i not in self.metrics:
                self.get_comparison_metrics(i)
            metrics = self.metrics[i]
            
            actual_key = f'actual_{metric_type}'
            optimal_key = f'optimal_{metric_type}'
            
            for actual, optimal in zip(metrics[actual_key], metrics[optimal_key]):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)

        def create_plot(fig):
            ax = fig.add_subplot(111)
            
            # Create DataFrame for seaborn
            data = pd.DataFrame({
                f'Optimal Game {metric_type.replace("_", " ").title()}': optimal_data,
                f'Actual Game {metric_type.replace("_", " ").title()}': actual_data
            })

                # Save data to CSV if requested
            if save_data:
                data.to_csv(os.path.join(self.main_dir, f'{metric_type}.csv'), index=False)

            # Count occurrences for size
            count_data = data.groupby([f'Optimal Game {metric_type.replace("_", " ").title()}', 
                                     f'Actual Game {metric_type.replace("_", " ").title()}']).size().reset_index(name='Count')
            
            # Create scatter plot
            sns.scatterplot(data=count_data, 
                          x=f'Optimal Game {metric_type.replace("_", " ").title()}', 
                          y=f'Actual Game {metric_type.replace("_", " ").title()}', 
                          size='Count', sizes=sizes, legend=False, alpha=0.5, ax=ax)
            
            # Add regression line
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(optimal_data, actual_data)
                line = slope * np.array([min(optimal_data), max(optimal_data)]) + intercept
                ax.plot([min(optimal_data), max(optimal_data)], line, 'r-', 
                    label=f'Regression line (R² = {r_value**2:.3f})', color='green')
            except ValueError:
                print("ValueError: Cannot fit regression line")
            
            # Add diagonal reference line
            max_val = max(max(actual_data), max(optimal_data))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')
            
            #ax.set_title(f'Optimal vs Actual Game {metric_type.replace("_", " ").title()} Comparison')
            ax.legend()

        if save_pdf:
            self.save_plot(create_plot, f"{metric_type.replace('_', ' ').title()} Comparison - scatter")
        else:
            create_plot(plt.figure(figsize=(10, 10)))
            plt.show()

    def plot_comparison_histogram(self, metric_type: str, n_games: int = None, bar_width: float = 0.35, overlay: bool = True, save_pdf: bool = False, save_data: bool = False):
        """Create grouped or overlaid bar chart comparing optimal vs actual games distributions."""
        print(f"Plotting comparison histogram for {metric_type}...")

        if n_games is None:
            n_games = len(self.df)

        # Collect data
        actual_data = []
        optimal_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)
        for i in tqdm.tqdm(sampled_indices):
            if i not in self.metrics:
                self.get_comparison_metrics(i)
            metrics = self.metrics[i]
            
            actual_key = f'actual_{metric_type}'
            optimal_key = f'optimal_{metric_type}'
            
            for actual, optimal in zip(metrics[actual_key], metrics[optimal_key]):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)

        def create_plot(fig):
            ax = fig.add_subplot(111)

            # Build counts & all_values exactly as before
            optimal_counts = pd.Series(optimal_data).value_counts().sort_index()
            actual_counts = pd.Series(actual_data).value_counts().sort_index()
            all_values = sorted(set(optimal_counts.index) | set(actual_counts.index))
            x = np.arange(len(all_values))
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in all_values])

            optimal_heights = [optimal_counts.get(v, 0) for v in all_values]
            actual_heights = [actual_counts.get(v, 0)  for v in all_values]

            if overlay:
                # draw both bars at the same x positions
                ax.bar(x, optimal_heights,
                       width=bar_width,
                       label='Optimal',
                       alpha=0.5,
                       edgecolor='black',
                       linewidth=1.2)
                ax.bar(x, actual_heights,
                       width=bar_width,
                       label='Actual',
                       alpha=0.5,
                       edgecolor='black',
                       linewidth=1.2)
            else:
                # side-by-side as you had it
                ax.bar(x - bar_width/2, optimal_heights,
                       width=bar_width,
                       label='Optimal',
                       alpha=0.7,
                       edgecolor='black',
                       linewidth=1.5)
                ax.bar(x + bar_width/2, actual_heights,
                       width=bar_width,
                       label='Actual',
                       alpha=0.7,
                       edgecolor='black',
                       linewidth=1.5)

            # add statistics text, labels, legend, etc. (unchanged)
            cohen_distance = WordleAnalyzer.cohen_d(actual_data, optimal_data)
            p_value        = ttest_rel(actual_data, optimal_data).pvalue
            ax.text(0.5, 1.03,
                    f"Cohen's d: {cohen_distance:.3g}, p-value: {p_value:.3g}",
                    ha='center', va='bottom', transform=ax.transAxes)

            ax.set_xlabel(metric_type.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.legend(title="Game Type")

            if save_data:
                pd.DataFrame({
                    'Value': all_values,
                    'Optimal_Count': optimal_heights,
                    'Actual_Count': actual_heights
                }).to_csv(
                    os.path.join(self.main_dir, f'{metric_type}_histogram.csv'),
                    index=False)

        if save_pdf:
            self.save_plot(create_plot,
                           f"{metric_type.replace('_', ' ').title()} Comparison - {'overlayed' if overlay else 'side-by-side'} histogram")
        else:
            create_plot(plt.figure(figsize=(10, 8)))
            plt.tight_layout()
            plt.show()

    def plot_comparison_histogram_0(self, metric_type: str, n_games: int = None, bar_width: float = 0.35, save_pdf: bool = False, save_data: bool = False, overlay: bool = False):
        """Create grouped bar chart comparing optimal vs actual games distributions."""
        if n_games is None:
            n_games = len(self.df)

        actual_data = []
        optimal_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)

        # Collect data
        for i in tqdm.tqdm(sampled_indices):
            if i not in self.metrics:
                self.get_comparison_metrics(i)
            metrics = self.metrics[i]
            
            actual_key = f'actual_{metric_type}'
            optimal_key = f'optimal_{metric_type}'
            
            for actual, optimal in zip(metrics[actual_key], metrics[optimal_key]):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)

        def create_plot(fig):
            ax = fig.add_subplot(111)
            
            # Create DataFrames for counting occurrences
            optimal_counts = pd.Series(optimal_data).value_counts().sort_index()
            actual_counts = pd.Series(actual_data).value_counts().sort_index()
            
            # Get all unique x values
            all_values = sorted(set(optimal_counts.index) | set(actual_counts.index))
            
            # Create x positions for bars
            x = np.arange(len(all_values))
            ax.set_xticks(x)  # Set x-ticks based on the actual values
            ax.set_xticklabels([f'{val}' for val in all_values]) 
            
            # Fill in missing values with zeros
            optimal_heights = [optimal_counts.get(val, 0) for val in all_values]
            actual_heights = [actual_counts.get(val, 0) for val in all_values]
            
            # Create bars
            optimal_bars = ax.bar(x - bar_width / 2, optimal_heights, bar_width,
                                label='Optimal', alpha=0.7, edgecolor='black', linewidth=1.5)
            actual_bars = ax.bar(x + bar_width / 2, actual_heights, bar_width,
                                label='Actual', alpha=0.7, edgecolor='black', linewidth=1.5)

            # Customize plot
            # Calculate and display Cohen's d and p-value
            cohen_distance = WordleAnalyzer.cohen_d(actual_data, optimal_data)
            p_value = ttest_rel(actual_data, optimal_data).pvalue
            
            # Add Cohen's d and p-value to the plot as text
            plt.text(0.5, 1, f"Cohen's d: {cohen_distance:.{3}g}, p-value: {p_value:.{3}g}", 
            horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10)
            
            ax.set_xlabel(f'{metric_type.replace("_", " ").title()}')
            ax.set_ylabel('Count')
            ax.legend(title="Game Type")

            # Save data if requested
            if save_data:
                data = pd.DataFrame({
                    'Value': all_values,
                    'Optimal_Count': optimal_heights,
                    'Actual_Count': actual_heights
                })
                data.to_csv(os.path.join(self.main_dir, f'{metric_type}_histogram.csv'), index=False)

        if save_pdf:
            self.save_plot(create_plot, f"{metric_type.replace('_', ' ').title()} Comparison - histogram")
        else:
            create_plot(plt.figure(figsize=(10, 10)))
            plt.show()

    def plot_density_heatmap(self, metric_type: str, n_games: int = None, bins: int = 20, save_pdf: bool = False):
        """Create a heatmap showing the density of points with option to save as PDF."""
        if n_games is None:
            n_games = len(self.df)
            
        actual_data = []
        optimal_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)
        
        for i in tqdm.tqdm(sampled_indices):
            if i not in self.metrics:
                self.get_comparison_metrics(i)
            metrics = self.metrics[i]
            actual_values = metrics[f'actual_{metric_type}']
            optimal_values = metrics[f'optimal_{metric_type}']
            
            for actual, optimal in zip(actual_values, optimal_values):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)

        def create_plot(fig):
            ax = fig.add_subplot(111)
            
            hist, xedges, yedges = np.histogram2d(optimal_data, actual_data, bins=bins)
            im = ax.imshow(hist.T, origin='lower', 
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                          aspect='auto', cmap='YlOrRd')
            
            fig.colorbar(im, ax=ax, label='Count')
            
            max_val = max(max(actual_data), max(optimal_data))
            min_val = min(min(actual_data), min(optimal_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
            
            ax.set_ylabel(f'Actual Game {metric_type.replace("_", " ").title()}')
            ax.set_xlabel(f'Optimal Game {metric_type.replace("_", " ").title()}')
            #ax.set_title(f'Density Heatmap: Optimal vs Actual Game {metric_type.replace("_", " ").title()}')

        if save_pdf:
            self.save_plot(create_plot, f'Density Heatmap: {metric_type.replace("_", " ").title()}')
        else:
            create_plot(plt.figure(figsize=(12, 10)))
            plt.show()

    def create_density_table(self, metric_type: str, n_games: int = None, bins: int = 10):
        """
        Create a density table showing the count of points at each x,y coordinate bin.
        
        Parameters:
        -----------
        metric_type: str
            'levenshtein', 'syllables', 'shared_chars', or 'glove_distance'
        n_games: int, optional
            Number of games to analyze
        bins: int, optional
            Number of bins for both x and y axes
        
        Returns:
        --------
        pandas.DataFrame
            A table showing the count of points in each bin
        """
        if n_games is None:
            n_games = len(self.df)
            
        actual_data = []
        optimal_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)
        
        for i in tqdm.tqdm(sampled_indices):
            if i not in self.metrics:
                self.get_comparison_metrics(i)
            metrics = self.metrics[i]
            actual_key = f'actual_{metric_type}'
            optimal_key = f'optimal_{metric_type}'
            
            actual_values = metrics[actual_key]
            optimal_values = metrics[optimal_key]
            
            for actual, optimal in zip(actual_values, optimal_values):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)
        
        # Create bins
        max_val = max(max(actual_data), max(optimal_data))
        min_val = min(min(actual_data), min(optimal_data))
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(optimal_data, actual_data, 
                                            bins=[bin_edges, bin_edges])
        
        # Create DataFrame with bin centers as index/columns
        density_df = pd.DataFrame(hist, 
                                index=pd.IntervalIndex.from_arrays(bin_edges[:-1], 
                                                                bin_edges[1:], 
                                                                closed='right'),
                                columns=pd.IntervalIndex.from_arrays(bin_edges[:-1], 
                                                                bin_edges[1:], 
                                                                closed='right'))
        
        # Add row and column totals
        density_df['Row Total'] = density_df.sum(axis=1)
        density_df.loc['Column Total'] = density_df.sum()
        #save density_df to csv
        density_df.to_csv(f'{self.main_dir}\\density_table_{metric_type}.csv')
        return density_df
        
    def get_popular_first_guesses(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get the most popular first guesses."""
        first_guesses = [guesses[0] for guesses in self.df['wordle_guesses']]
        return pd.Series(first_guesses).value_counts().head(top_n).items()
    
    def get_hard_mode_stats(self) -> Dict[str, float]:
        """Compare performance between hard mode and normal mode."""
        hard_mode_stats = self.df.groupby('hard_mode')['num_guesses'].agg(['mean', 'count']).to_dict('index')
        return {
            'hard_mode_avg': hard_mode_stats.get(True, {'mean': 0})['mean'],
            'normal_mode_avg': hard_mode_stats.get(False, {'mean': 0})['mean'],
            'hard_mode_games': hard_mode_stats.get(True, {'count': 0})['count'],
            'normal_mode_games': hard_mode_stats.get(False, {'count': 0})['count']
        }
    
    def get_average_guesses(self) -> float:
        """Calculate the average number of guesses across all games."""
        return self.df['num_guesses'].mean()
    
    def get_guess_distribution(self) -> Dict[int, int]:
        """Get the distribution of number of guesses."""
        return self.df['num_guesses'].value_counts().sort_index().to_dict()
    
    def plot_guess_distribution(self, save_pdf: bool = False):
        """Plot the distribution of number of guesses."""
        def create_plot(fig):
            ax = fig.add_subplot(111)
            
            ax.bar(self.get_guess_distribution().keys(), self.get_guess_distribution().values(), alpha=0.7)
            
            ax.set_xlabel('Number of Guesses')
            ax.set_ylabel('Number of Games')
        
        if save_pdf:
            self.save_plot(create_plot, "Guess Distribution")
        else:
            create_plot(plt.figure(figsize=(12, 10)))
            plt.show()

    @staticmethod
    def cohen_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)    
    
    def analyze_optimal_choices(self, n_games: int = None, save_pdf: bool = False) -> pd.DataFrame:
        """
        Analyze how often players chose the optimal word for each guess position.
        
        Parameters:
        -----------
        n_games: int, optional
            Number of games to analyze. If None, analyzes all games.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing statistics about optimal word choices
        """
        if n_games is None:
            n_games = len(self.df)
        
        optimal_choices_data = []
        sampled_indices = np.random.choice(self.df.index, size=min(n_games, len(self.df)), replace=False)
        
        # Collect data
        for i in tqdm.tqdm(sampled_indices):
            row = self.df.iloc[i]
            player_guesses = row['wordle_guesses']
            optimal_sequence = row['optimal']
            
            # For each guess position (except the first), check if player chose optimal word
            for guess_idx in range(len(optimal_sequence)):
                optimal_word = optimal_sequence[guess_idx][1][0].lower()  # Ensure lowercase
                player_word = player_guesses[guess_idx + 1].lower()
                
                optimal_choices_data.append({
                    'game_id': i,
                    'guess_position': guess_idx + 2,  # +2 because we're looking at next guess
                    'player_word': player_word,
                    'optimal_word': optimal_word,
                    'chose_optimal': player_word == optimal_word if player_word is not None else False
                })
        
        results_df = pd.DataFrame(optimal_choices_data)
        
        # Calculate summary statistics
        summary_stats = (results_df
                        .groupby('guess_position')
                        .agg({
                            'chose_optimal': ['count', 'sum', 'mean'],
                            'game_id': 'nunique'
                        })
                        .round(3))
        
        summary_stats.columns = ['total_guesses', 'optimal_choices', 
                            'optimal_percentage', 'unique_games']
        
        # Create visualization
        def create_plot(fig):
            ax = fig.add_subplot(111)
            
            ax.bar(summary_stats.index, summary_stats['optimal_percentage'],
                    alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Guess Position')
            ax.set_ylabel('Percentage of Optimal Choices')
        
        # Add percentage labels on top of bars
            for i, v in enumerate(summary_stats['optimal_percentage']):
                ax.text(i+2, v + 0.01, f'{v:.1%}', 
                        ha='center', va='bottom')  # Set the label to the center and above the bar
        
        if save_pdf:
            self.save_plot(create_plot, f"Optimal Word Choice Percentage by Guess Position")
        else:
            create_plot(plt.figure(figsize=(12, 10)))
            plt.show()
        
        return results_df, summary_stats


# Example usage
def main():
    # Initialize the analyzer with your CSV file
    analyzer = WordleAnalyzer(r'data_analysis\data\merged_data.csv',\
                load_pickle=True, avg_dec_place=2, specific_dec_places={"actual_avg_word2vec_distance": 1, "optimal_avg_word2vec_distance": 1,"actual_avg_glove_distance": 1, \
                                                                        "optimal_avg_glove_distance": 1, "actual_avg_shared_chars": 0, "optimal_avg_shared_chars": 0, \
                                                                        "actual_avg_levenshtein": 0, "optimal_avg_levenshtein": 0})
    analyzer.avg_func = "mean"
    # Get basic statistics
    print(f"Average guesses: {analyzer.get_average_guesses():.2f}")
    
    # Get and print guess distribution
    distribution = analyzer.get_guess_distribution()
    print("\nGuess Distribution:")
    for guesses, count in distribution.items():
        print(f"{guesses} guesses: {count} games")
    
    # Get popular first guesses
    print("\nMost Popular First Guesses:")
    for guess, count in analyzer.get_popular_first_guesses():
        print(f"{guess}: {count} times")
    
    # Get hard mode statistics
    hard_mode_stats = analyzer.get_hard_mode_stats()
    print("\nHard Mode vs Normal Mode:")
    print(f"Hard Mode Average: {hard_mode_stats['hard_mode_avg']:.2f} ({hard_mode_stats['hard_mode_games']} games)")
    print(f"Normal Mode Average: {hard_mode_stats['normal_mode_avg']:.2f} ({hard_mode_stats['normal_mode_games']} games)")
    print("\n\n")
    #analyzer.plot_comparison_histogram('avg_glove_distance', save_pdf=True, save_data=True)
    #analyzer.dump_pickle_metrics() 
    
    # Create visualizations
    analyzer.plot_comparison_histogram('avg_levenshtein', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_shared_chars', save_pdf=True, save_data=True)
    '''
    analyzer.plot_guess_distribution(save_pdf=True)
    analyzer.plot_comparison_scatter('levenshtein', save_pdf=True, save_data=True)
    analyzer.dump_pickle_metrics() # TODO REMOVE THIS WHEN LOADING PICKLE
    analyzer.plot_comparison_scatter('avg_levenshtein', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('levenshtein', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_levenshtein', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('syllables', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_syllables', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('syllables', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_syllables', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('shared_chars', save_pdf=True, save_data=True)  
    analyzer.plot_comparison_scatter('avg_shared_chars', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('shared_chars', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_shared_chars', save_pdf=True, save_data=True)
    
    analyzer.plot_comparison_scatter('glove_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_glove_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('glove_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_glove_distance', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('word2vec_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_word2vec_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('word2vec_distance', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_word2vec_distance', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('rhyme0_count', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_rhyme0_count', save_pdf=True, save_data=True)  
    analyzer.plot_comparison_histogram('rhyme0_count', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_rhyme0_count', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('rhyme1_count', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_rhyme1_count', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('rhyme1_count', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_rhyme1_count', save_pdf=True, save_data=True)

    analyzer.plot_comparison_scatter('gpt_tokens', save_pdf=True, save_data=True)
    analyzer.plot_comparison_scatter('avg_gpt_tokens', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('gpt_tokens', save_pdf=True, save_data=True)
    analyzer.plot_comparison_histogram('avg_gpt_tokens', save_pdf=True, save_data=True)
    
    print(analyzer.analyze_optimal_choices(save_pdf=True)[1])'''

if __name__ == "__main__":
    main()


## how many guesses did optimal take vs how many guesses did the player take
## how many time they chose optimal words