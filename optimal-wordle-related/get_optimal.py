from doddle.cli import run, my_run
from doddle.words import Word
from doddle.enums import SolverType
import pandas as pd
import sys

# command line call in format: python3 get_optimal.py <file_name.csv>

def get_data(file):
    data = pd.read_csv(file)
    return data

def get_optimal(file):
    data = get_data(file)
    # initialize a new data column called optimal
    data['optimal'] = None

    for index, row in data.iterrows():
        # pair_wise_optimal = []
        # for i in range(len(wordle_guesses) - 1):
        #     optimal_guess = my_run(Word(wordle_answer), wordle_guesses[:i+1], solver_type=solver)
        #     pair_wise_optimal.append(optimal_guess)
        optimal = []
        wordle_answer = row['wordle_answer']
        wordle_guesses = eval(row['wordle_guesses'])
        # print(wordle_answer)
        # print(wordle_guesses)

        for i in range(len(wordle_guesses) - 1):
            optimal_guess = my_run(Word(wordle_answer), wordle_guesses[:i+1], solver_type=SolverType.ENTROPY)
            optimal.append(optimal_guess) 
            # print(optimal)
        
        data.at[index, 'optimal'] = optimal
    # print('The data has been successfully processed')
    return data

# make it so that when the file is ran, the get_optimal function is called
if __name__ == "__main__":
    file = sys.argv[1]
    output = file.replace('.csv', '') + '_output.csv'
    try:
        new_data = get_optimal(file)
        new_data.to_csv(output, index=False)
        print('The data has been successfully processed')
    except FileNotFoundError:
        print('File not found')
    except Exception as e:
        print(e)
        print('An error occurred')