import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Union
from ast import literal_eval
from nltk.tokenize import SyllableTokenizer
import numpy as np
from collections import defaultdict
import tqdm

class WordleAnalyzer:
    tokenizer = SyllableTokenizer()

    def __init__(self, csv_path: str):
        """Initialize the WordleAnalyzer with a CSV file path."""
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Convert string representations of lists to actual lists
        self.df['wordle_guesses'] = self.df['wordle_guesses'].apply(literal_eval)
        self.df['optimal'] = self.df['optimal'].apply(literal_eval)
    
    def get_average_guesses(self) -> float:
        """Calculate the average number of guesses across all games."""
        return self.df['num_guesses'].mean()
    
    def get_guess_distribution(self) -> Dict[int, int]:
        """Get the distribution of number of guesses."""
        return self.df['num_guesses'].value_counts().sort_index().to_dict()
    
    def plot_guess_distribution(self):
        """Plot the distribution of number of guesses."""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='num_guesses')
        plt.title('Distribution of Number of Guesses')
        plt.xlabel('Number of Guesses')
        plt.ylabel('Count')
        plt.show()
    
    @staticmethod
    def levenshtein_between_guesses(source: str, target: str) -> int:
        """Calculate Levenshtein distance between two words."""
        if len(source) == 0:
            return len(target)
        if len(target) == 0:
            return len(source)
        if source[0] == target[0]:
            return WordleAnalyzer.levenshtein_between_guesses(source[1:], target[1:])
        direct_edit = WordleAnalyzer.levenshtein_between_guesses(source[1:], target[1:])
        insert = WordleAnalyzer.levenshtein_between_guesses(source, target[1:])
        delete = WordleAnalyzer.levenshtein_between_guesses(source[1:], target)
        return 1 + min(delete, min(direct_edit, insert))
    
    @staticmethod
    def avg_levenshtein_within_game(guess_list: List[str], start_idx: str) -> Union[float, str]:
        """Calculate average Levenshtein distance within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            return "no distance"
        total_distance = 0
        comp = 0
        for i in range(0, start_idx):
            total_distance += WordleAnalyzer.levenshtein_between_guesses(guess_list[i], guess_list[i+1])
            comp += 1
        return total_distance / comp if comp > 0 else 0
    
    @staticmethod
    def common_syllables(word1: str, word2: str) -> int:
        """Calculate number of common syllables between two words."""
        syllables1 = set(WordleAnalyzer.tokenizer.tokenize(word1))
        syllables2 = set(WordleAnalyzer.tokenizer.tokenize(word2))
        return len(syllables1.intersection(syllables2))
    
    @staticmethod
    def avg_common_syllables_within(guess_list: List[str], start_idx: str) -> Union[float, str]:
        """Calculate average number of common syllables within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            return "no common syllables"
        total_common_syllables = 0
        comp = 0
        for i in range(0, start_idx):
            total_common_syllables += WordleAnalyzer.common_syllables(guess_list[i], guess_list[i+1])
            comp += 1
        return total_common_syllables / comp if comp > 0 else 0
    
    @staticmethod
    def shared_chars(word1: str, word2: str) -> int:
        """Calculate number of shared characters between two words."""
        return len(set(word1).intersection(set(word2)))
    
    @staticmethod
    def avg_shared_chars_within(guess_list: List[str], start_idx: str) -> Union[float, str]:
        """Calculate average number of shared characters within a game's guesses. from guess i to guess 0 inclusive"""
        if len(guess_list) == 1:
            return "no shared characters"
        total_shared_chars = 0
        comp = 0
        for i in range(0, start_idx):
            total_shared_chars += WordleAnalyzer.shared_chars(guess_list[i], guess_list[i+1])
            comp += 1
        return total_shared_chars / comp if comp > 0 else 0
    
    def get_optimal_guesses(self, game_index: int) -> List[str]:
        """Get optimal guesses for game at specified index."""
        row = self.df.iloc[game_index]
        actual_guesses = row['wordle_guesses']
        games = []
        for i in range():
            pass

    def get_comparison_metrics(self, game_index: int) -> Dict[str, List[float]]:
        """Get comparison metrics for actual vs optimal game at specified index."""
        row = self.df.iloc[game_index]
        actual_guesses = row['wordle_guesses']
        optimal_sequence = [guess_tuple[1][0] for guess_tuple in row['optimal']]
        metrics = {
            'actual_levenshtein': [],
            'optimal_levenshtein': [],
            'actual_syllables': [],
            'optimal_syllables': [],
            'actual_shared_chars': [],
            'optimal_shared_chars': []
        }
        
        # Calculate metrics for actual guesses
        for i in range(len(actual_guesses) - 1):
            metrics['actual_levenshtein'].append(
                self.levenshtein_between_guesses(actual_guesses[i], actual_guesses[i+1]))
            metrics['actual_syllables'].append(
                self.common_syllables(actual_guesses[i], actual_guesses[i+1]))
            metrics['actual_shared_chars'].append(
                self.shared_chars(actual_guesses[i], actual_guesses[i+1]))
        
        # Calculate metrics for optimal guesses
        for i in range(len(optimal_sequence) - 1):
            metrics['optimal_levenshtein'].append(
                self.levenshtein_between_guesses(optimal_sequence[i], optimal_sequence[i+1]))
            metrics['optimal_syllables'].append(
                self.common_syllables(optimal_sequence[i], optimal_sequence[i+1]))
            metrics['optimal_shared_chars'].append(
                self.shared_chars(optimal_sequence[i], optimal_sequence[i+1]))
        
        return metrics
    
    def plot_comparison_scatter(self, metric_type: str, n_games: int = None):
        """
        Create scatter plot comparing actual vs optimal games for specified metric.
        
        metric_type: 'levenshtein', 'syllables', or 'shared_chars'
        n_games: number of games to analyze
        """

        if n_games is None:
            n_games = len(self.df)

        actual_data = []
        optimal_data = []
        
        for i in tqdm.tqdm(range(min(n_games, len(self.df)))):
            metrics = self.get_comparison_metrics(i)
            actual_key = f'actual_{metric_type}'
            optimal_key = f'optimal_{metric_type}'
            
            # Extend the shorter list to match the longer one's length
            max_len = max(len(metrics[actual_key]), len(metrics[optimal_key]))
            actual_values = metrics[actual_key] + [None] * (max_len - len(metrics[actual_key]))
            optimal_values = metrics[optimal_key] + [None] * (max_len - len(metrics[optimal_key]))
            
            for actual, optimal in zip(actual_values, optimal_values):
                if actual is not None and optimal is not None:
                    actual_data.append(actual)
                    optimal_data.append(optimal)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(actual_data, optimal_data, alpha=0.5)
        plt.xlabel(f'Actual Game {metric_type.replace("_", " ").title()}')
        plt.ylabel(f'Optimal Game {metric_type.replace("_", " ").title()}')
        plt.title(f'Actual vs Optimal Game {metric_type.replace("_", " ").title()} Comparison')
        
        # Add diagonal line for reference
        max_val = max(max(actual_data), max(optimal_data))
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    
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
    def save_plot_to_pdf(self, fig, filename):
        fig.savefig(filename, bbox_inches='tight')

# Example usage
def main():
    # Initialize the analyzer with your CSV file
    analyzer = WordleAnalyzer(r'C:\Users\adamk\Documents\wordle_research\wordle-research\data_analysis\data\merged_data.csv')
    # print optimal col of first row
    print(analyzer.df['optimal'][0])
    # Get basic statistics
    print(f"Average guesses: {analyzer.get_average_guesses():.2f}")
    
    # Get and print guess distribution
    distribution = analyzer.get_guess_distribution()
    print("\nGuess Distribution:")
    for guesses, count in distribution.items():
        print(f"{guesses} guesses: {count} games")
    
    # Analyze optimal vs actual performance
    #optimal_analysis = analyzer.analyze_optimal_vs_actual()
    #print(f"\nAverage excess guesses compared to optimal: {optimal_analysis['average_excess_guesses']:.2f}")
    
    # Get popular first guesses
    print("\nMost Popular First Guesses:")
    for guess, count in analyzer.get_popular_first_guesses():
        print(f"{guess}: {count} times")
    
    # Get hard mode statistics
    hard_mode_stats = analyzer.get_hard_mode_stats()
    print("\nHard Mode vs Normal Mode:")
    print(f"Hard Mode Average: {hard_mode_stats['hard_mode_avg']:.2f} ({hard_mode_stats['hard_mode_games']} games)")
    print(f"Normal Mode Average: {hard_mode_stats['normal_mode_avg']:.2f} ({hard_mode_stats['normal_mode_games']} games)")
    print(analyzer.get_comparison_metrics(0))
    print(analyzer.avg_levenshtein_within_game(['world', 'leafs', 'clang', 'bantu', 'banal'], 2)) 

    # Create visualizations
    #analyzer.plot_guess_distribution()
    #analyzer.plot_comparison_scatter('levenshtein')
   # analyzer.plot_comparison_scatter('syllables')
    analyzer.plot_comparison_scatter('shared_chars')
# ['world', 'leafs', 'clang', 'bantu', 'banal']


if __name__ == "__main__":
    main()


## how many guesses did optimal take vs how many guesses did the player take
## how many time they chose optimal words