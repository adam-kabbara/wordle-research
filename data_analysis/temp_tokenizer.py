'''
This is a temp/test file and was used to test different types of tokenizers.
The result was that the nltk syllable tokenizer was the best for the purpose of the project.
'''

from nltk.tokenize import SyllableTokenizer
from nltk.stem import WordNetLemmatizer
from spacy.cli.download import download
from transformers import GPT2Tokenizer
import re
import pronouncing

###################################################
def syllable_tokenize(text):
    syllable_tokenizer = SyllableTokenizer()
    return syllable_tokenizer.tokenize(text)

def pre_trained_bpe_tokenize(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer.tokenize(text)
###################################################

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

def rhyme_tokenize_1(text1, text2):
    ending1 = get_phonetic_ending(text1)
    ending2 = get_phonetic_ending(text2)
    
    if ending1 and ending2:
        return ending1 == ending2  # Check if the phonetic endings match
    else:
        return False
    
###################################################

def rhyme_tokenize_0(text1, text2):
    return text2 in pronouncing.rhymes(text) or text1 in pronouncing.rhymes(text2)

def get_syllables(word):
    phones = pronouncing.phones_for_word(word)
    
    if phones:
        # Get the ARPAbet phonetic representation of the word
        phone_string = phones[0]
        
        # Split phonetic transcription into individual phonemes
        phonemes = phone_string.split()
        
        # Vowel phonemes in ARPAbet (they end with 0, 1, or 2)
        vowels = [p for p in phonemes if p[-1].isdigit()]
        
        # Get the syllables
        syllable_count = pronouncing.syllable_count(phone_string)
        
        # For now, a rough estimate of syllables by dividing the word into the same number of parts as vowel sounds
        # This is a heuristic, and might not work perfectly for all words
        syllables = []
        step = len(word) // syllable_count if syllable_count > 0 else len(word)
        current_pos = 0
        
        for i in range(syllable_count):
            next_pos = current_pos + step
            if i == syllable_count - 1:
                syllables.append(word[current_pos:])  # Add the remaining part of the word as the last syllable
            else:
                syllables.append(word[current_pos:next_pos])
            current_pos = next_pos
        
        return vowels, syllables
    else:
        return [], []
    
###################################################

def morphological_tokenize(text):
    prefixes = r'(un|dis|re|pre|mis|in|im|non|over|under|sub|super|trans)'
    suffixes = r'(able|ible|ing|ed|er|est|ly|ness|s|es|ful|less|ment|tion|al|ism)'
    
    prefix_match = re.match(prefixes, text)
    prefix = prefix_match.group(0) if prefix_match else ''
    
    # Match the suffix
    suffix_match = re.search(suffixes + r'$', text)
    suffix = suffix_match.group(0) if suffix_match else ''
    
    # Extract the root by removing the prefix and suffix
    root = text
    if prefix:
        root = root[len(prefix):]
    if suffix:
        root = root[:len(root) - len(suffix)]
    
    # Return the parts of the word
    return prefix, root, suffix



#########################################
def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(text, pos='v')
    return text.split(lemma).insert(1, lemma)

def _morphological_tokenize(text, nlp):
    doc = nlp(text)
    return [token.text for token in doc]

def download_resources():
    download(model='en_core_web_sm')
#########################################

if __name__ == '__main__':
    #download_resources()
    #nlp = spacy.load('en_core_web_sm')

    text = "hello"
    print(syllable_tokenize(text))
    print(get_syllables(text))
    print(rhyme_tokenize_0(text, "mellow"))
    print(rhyme_tokenize_1(text, "mellow"))
    #print(pre_trained_bpe_tokenize(text))
    #print(morphological_tokenize(text, nlp)) bad
    #print(morphological_tokenize(text))
