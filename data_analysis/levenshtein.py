import csv
import ast

def levenshtein_between_guesses(source,target):
    if (len(source)==0):
        return len(target)
    if (len(target)==0):
        return len(source)
    if (source[0]==target[0]):
        return levenshtein_between_guesses(source[1:],target[1:])
    direct_edit=levenshtein_between_guesses(source[1:],target[1:])
    insert=levenshtein_between_guesses(source,target[1:]) # insert the same alphabit as the start of target
    delete=levenshtein_between_guesses(source[1:],target) # delete the starting alphabit of source
    return 1+ min(delete,min(direct_edit,insert))

def avg_levenshtein_within_game(guess_list):
    if (len(guess_list)==1):
        return "no distance"
    comp=0
    total_distance=0
    for i in range (len(guess_list)-1):
        # only considering distances between the guess and the next guess as they can potentially be similar due to solidified thinking
        # the initial guess and the final guess are EXPECTED to be really apart and it doesn't show anything
        total_distance+=levenshtein_between_guesses(guess_list[i],guess_list[i+1]) 
        comp+=1
    return total_distance/comp

def avg_game_levenshtein(list_of_games):
    games=0
    distance=0
    for game in list_of_games:
        if (len(game)==1):
            continue
        increment=avg_levenshtein_within_game(game)
        distance+=increment
        games+=1
    return distance/games

def import_from_csv():
    file = open("data_analysis/data_1.csv") # Put the name of the data file. Note: it shall be in the same folder as this file
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    array_like_data = []
    non_hard_mode_guesses=[]
    hard_mode_guesses=[]
    for row in csvreader:
        if (row[6]=='false'):
            list = ast.literal_eval(row[1]) # change the guesses from string to list
            non_hard_mode_guesses.append(list)
        else:
            list = ast.literal_eval(row[1]) # change the guesses from string to list
            hard_mode_guesses.append(list)
        array_like_data.append(row)
    file.close()
    return array_like_data,non_hard_mode_guesses,hard_mode_guesses

def test():
    list2=[['bantu', 'banal']]
    return avg_game_levenshtein(list2)

#_,non_hard,hard=import_from_csv()

print(test())
# print(avg_game_levenshtein(non_hard)) # average Levenshtein distance 3.677
# print(avg_game_levenshtein(hard)) # average Levenshtein distance 3.506


