import csv
#from nltk.tokenize import word_tokenize
from somajo import SoMaJo
from someweta import ASPTagger
import re 
import emoji

DATA_PATH = './data/train.SemEval.csv'

def load_dataset(filename: str):
    """
    Function that reads the dataset and saves information about a
    tweet within a dictionary. 
    It contains: ID, Tweet, Classification, Rephrasing.

    Input:
    1. filename (str)   : Name of the file that is supposed to be read in.

    Output:
    1. dataset (list)   : List which listelements is a dictionary for each
                            Tweet in the given dataset. Each dictionary has
                            the following keys:
                            tweetID
                            tweetText
                            class
                            rephrase


    """
    dataset = list()

    with open(filename, "r", encoding="UTF8") as f:

        data = csv.reader(f, delimiter=",")

        # this just skips the first line
        next(data)  

        for row in data:
            tweet_dict = dict()
            tweet_dict["tweetID"] =row[0]
            tweet_dict["tweetText"] = row[1]
            tweet_dict["class"] = int(row[2])
            
            if tweet_dict["class"] == 1:
                tweet_dict["rephrase"] =row[3]
            else: 
                tweet_dict["rephrase"] = None

            dataset.append(tweet_dict)

    return dataset

def tokenizer   (dataset: list):
    """
    Function that tokenizes the text in a given dataset. 
    And additionally adds Part Of Speech Tags for each token.

    Input:
    1. dataset (list)   : A list which contains a dictionary.
                            Dictionary key "tweetText" contains the
                            text which is supposed to be tokenized.
    
    Output:
    1. dataset (list)   : The same dataset (list) that was given as input
                            but the key "tweetTokenized" has been added
                            which holds the tokenized sentence.
    """
    # create object for tokenizer 
    tokenizer = SoMaJo("en_PTB", split_sentences=False)

    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict 
        to_tokenize = [tweet["tweetText"]] # must be list for tokenizer 
        # tokenize the sentence
        tokenized_obj = tokenizer.tokenize_text(to_tokenize)
        for paragraph in tokenized_obj:
            tokenized_tweet = [token.text for token in paragraph]

        # create a new dict-element with the tokenized text
        tweet["tweetTokenized"] = tokenized_tweet

    return dataset


def pos_tagging(dataset):
    # load model
    model = "english_newspaper_2017-09-15.model"

    """
    # example sentences
    sentences = [["Ein", "Satz", "ist", "eine", "Liste", "von", "Tokens", "."],
             ["Zeitfliegen", "mögen", "einen", "Pfeil", "."]]
    """
    # create object for pos tagger
    asptagger = ASPTagger()
    asptagger.load(model)

    # go through dataset to get tokenized text
    for tweet in dataset:
        tokenized_tweet = [tweet[tokenized_tweet]]
        
        # take the sentence & pos tag it
        for sentence in tokenized_tweet:
            tagged_sentence = asptagger.tag_sentence(sentence)
            posTagged_tweet = tagged_sentence

            # create new dictionary key
            dataset["posTagged_tweet"] = posTagged_tweet

    return dataset

def cleanURLS(dataset: list):
    pass

dataset = load_dataset(DATA_PATH)
#dataset = load_dataset("train.En.csv")
dataset = tokenizer(dataset)

""" - don't delete yet, still needed script in there 
def load_dataset(filename: str):
    tweet_text = []  # stores only the text of the tweet as a string
    tweet_emojis = []  # stores the emojis of a tweet as a list, each emoji separated by comma
    sarcasm = []  # stores "0" or "1", depending on if tweet is sarcastic (1) or not (0)
    sarcastic = 0  # to count overall amount of sarcastic tweets
    unsarcastic = 0  # to count overall amount of non-sarcastic tweets
    emoji_counter = []  # count number of emojis per tweet
    with open(filename, "r", encoding="UTF8") as f:
        data = csv.reader(f, delimiter=",")
        next(data)  # this just skips the first line
        for row in data:
            tweet_text.append(
                word_tokenize(emoji.replace_emoji(row[1], replace='')))  # replace emojis with blank, replace this with
            # SoMaJo tokenizer instead of NLTK

            # extracting Unicode representations of emojis (demojize) and adding them to tweet_emojis
            tweet_emojis.append([emoji.demojize(entry['emoji']) for entry in emoji.emoji_list(row[1])])

            emoji_counter.append(emoji.emoji_count(row[1]))  # count number of emojis per tweet
            sarcasm.append(row[2])  # add label
            if row[2] == "1":
                # add text of unsarcastic rephrases to tweet_text
                tweet_text.append(word_tokenize(emoji.replace_emoji(row[3], replace='')))

                # extracting Unicode representations of emojis for unsarcastic rephrases of sarcastic tweets
                tweet_emojis.append([emoji.demojize(entry['emoji']) for entry in emoji.emoji_list(row[3])])

                emoji_counter.append(emoji.emoji_count(row[3]))
                sarcasm.append("0")
                sarcastic += 1
                unsarcastic += 1
            if row[2] == "0":
                unsarcastic += 1
    return tweet_text, tweet_emojis, emoji_counter, sarcasm, sarcastic, unsarcastic
"""
