'''
This is a temp/test file and was used to test different types of tokenizers.
The result was that the nltk syllable tokenizer was the best for the purpose of the project.
'''

from nltk.tokenize import SyllableTokenizer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.cli.download import download
from transformers import GPT2Tokenizer
import re

def syllable_tokenize(text):
    syllable_tokenizer = SyllableTokenizer()
    return syllable_tokenizer.tokenize(text)

def pre_trained_bpe_tokenize(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer.tokenize(text)

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

    text = "unspeakable"
    print(syllable_tokenize(text))
    #print(pre_trained_bpe_tokenize(text))
    #print(morphological_tokenize(text, nlp)) bad
    #print(morphological_tokenize(text))
