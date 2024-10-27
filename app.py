import warnings
from tqdm import tqdm

warnings.simplefilter('ignore')
import time
from collections import OrderedDict

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import nltk
nltk.download('punkt')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time

from sklearn.manifold import TSNE

# Suppress warnings within the code
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def preprocess_string(s):
    """
    Preprocesses a string by removing non-word characters and replacing digits with empty space.
    """
    s = re.sub(r"[^\w\s]", '', s)  # Remove non-word characters
    s = re.sub(r"\s+", '', s)     # Replace all runs of whitespaces with no space
    s = re.sub(r"\d", '', s)      # replace digits with no space
    return s.lower()             # Convert to lowercase


def preprocess(words):
    """
    Preprocesses a list of words by tokenizing, removing punctuation, and converting to lowercase.
    """
    tokens = nltk.word_tokenize(words)
    tokens = [preprocess_string(w) for w in tokens]
    return [w.lower() for w in tokens if len(w) != 0 and not (w in string.punctuation)]
    return(tokens)


def make_predictions(my_words, freq_grams, normlize=1, vocabulary=None):
    """
    Generates predictions for the conditional probability of the next word given a sequence.

    Args:
        my_words (list): A list of words in the input sequence.
        freq_grams (dict): A dictionary containing frequency of n-grams.
        normlize (int, optional): A normalization factor for calculating probabilities. Defaults to 1.
        vocabulary (list, optional): A list of words in the vocabulary. Defaults to None.

    Returns:
        list: A list of predicted words along with their probabilities, sorted in descending order.
    """

    vocab_probabilities = {}  # Initialize a dictionary to store predicted word probabilities
    context_size = len(list(freq_grams.keys())[0])  # Determine the context size from n-grams keys

    # Preprocess input words and take only the relevant context words
    my_tokens = preprocess(my_words)[0:context_size - 1]

    # Calculate probabilities for each word in the vocabulary given the context
    for next_word in (vocabulary or []):
        temp = my_tokens.copy()
        temp.append(next_word)  # Add the next word to the context

        # Calculate the conditional probability using the frequency information
        if normlize != 0:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] / normlize
        else:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)]

    # Sort the predicted words based on their probabilities in descending order
    vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)

    return vocab_probabilities  # Return the sorted list of predicted words and their probabilities


def write_song(model, vocab, tokens, index_to_token, CONTEXT_SIZE=4, number_of_words=100, device='cpu'):
    """
    Generates a song by sampling words from the provided model.

    Args:
        model (torch.nn.Module): The language model used to predict the next word.
        vocab (function): A function that converts words into token IDs.
        tokens (list): A list of initial tokens to seed the model.
        index_to_token (dict): A dictionary mapping token IDs to words.
        CONTEXT_SIZE (int): The number of previous words used to predict the next one.
        number_of_words (int): The number of words to generate.
        device (str): The device to perform computations on.

    Returns:
        str: The generated song lyrics.
    """
    my_song = ("We are no strangers to love\n"
               "You know the rules and so do I\n"
               "A full commitment's what I'm thinking of\n"
               "You wouldn't get this from any other guy\n")

    for i in range(number_of_words):
        with torch.no_grad():
            # Prepare context with the last CONTEXT_SIZE tokens
            context = torch.tensor(
                [vocab(tokens[i - j - 1]) for j in range(CONTEXT_SIZE)]
            ).to(device)
            word_inx = torch.argmax(model(context.unsqueeze(0)), dim=1)
            my_song += " " + index_to_token[word_inx.item()]

    return my_song

    my_song = ""
    for i in range(number_of_words):
        with torch.no_grad():
            context = torch.tensor(vocab([tokens[i - j - 1] for j in range(CONTEXT_SIZE)])).to(device)
            word_inx = torch.argmax(model(context))
            my_song += " " + index_to_token[word_inx.detach().item()]

    return my_song


def train(dataloader, model, number_of_epochs=100, show=10):
    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch+1}/{number_of_epochs}")
        for batch in dataloader:
            # Training logic here
            print(f"Processed batch: {batch}")
