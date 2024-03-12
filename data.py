import csv
#from nltk.tokenize import word_tokenize
from somajo import SoMaJo
from someweta import ASPTagger
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import emoji
import html

DATA_PATH = './data/train.En__1_.csv'

analyzer = SentimentIntensityAnalyzer()

def load_dataset(filename: str):
    """
    Function that reads the dataset, converts HTML encoded text to plain text, 
    and saves information about a tweet within a dictionary.

    Input:
    1. filename (str)   :   Name of the file that is supposed to be read in.

    Output:
    1. dataset (list)   :   List of dictionaries where each element is a
                            Tweet from the given dataset. Each dictionary has
                            the following keys:
                                tweetID     :   numeric ID of the tweet
                                tweetText   :   text of the tweet formatted as a string
                                class       :   classification of the tweet as 
                                                non-sarcastic (0) or sarcastic (1)
                            More keys are added in the following steps.
                            Sarcastic rephrases are handled as individual tweets.
    """
    dataset = list()

    with open(filename, "r", encoding="UTF8") as f:

        data = csv.reader(f, delimiter=",")

        # this just skips the first line
        next(data)

        for row in data:
            # this converts html encoded text to plain text (solves &amp problem)
            decoded_row = [html.unescape(cell) for cell in row]
            tweet_dict = dict()
            tweet_dict["tweetID"] = decoded_row[0]
            tweet_dict["tweetText"] = decoded_row[1]
            tweet_dict["class"] = int(decoded_row[2])

            dataset.append(tweet_dict)

            if tweet_dict["class"] == 1: 
                #rephrases get their own dictionary entry
                #same ID as original tweets, we still know what the rephrase/OG tweet is bc of class
                rephrase_dict = dict()
                rephrase_dict["tweetID"] = decoded_row[0]
                rephrase_dict["tweetText"] = decoded_row[3]
                rephrase_dict["class"] = 0
                dataset.append(rephrase_dict)

    return dataset


def cleanURLS(dataset: list):
    """
    Function that removes URLs from our analysis.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            The dictionary key "tweetText" contains the
                            text to be modified for URL removal.

    Output:
    1. dataset (list)   :   Inputted dataset that contains tweet versions
                            that are completely free of URLs. A new key 
                            "tweetwithoutURL" has been added
                            accordingly.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        tweet_toclean = tweet["tweetText"]
        # remove URLs from text
        no_URL = url_pattern.sub('', tweet_toclean)
        # create a new dict-element with the clean text
        tweet["tweetWithoutURL"] = no_URL
    return dataset


def handling_emojis(dataset: list):
    """
    Function that separates and prepares emojis for further processing.

    Input:
    1. dataset (list)   :   A list of dictionaries. The dictionary key "tweetwithoutURL" contains the
                            text to be modified for emoji processing.
                        
    Output:
    1. dataset (list)   :   Inputted dataset with several added dictionary entries:
                                "tweetWithoutEmoji"         :   tweet formatted as a string without emojis
                                "tweetWithEmojiDescription" :   tweet formatted as a string with emoji description;
                                                                emojis are replaced by their description and delimited
                                                                by "<e>" and "</e." (e.g., <e> sad face </e>)
                                "emojis"                    :   list of all emojis that appear in the tweet;
                                                                each emoji is formatted as a string with emoji 
                                                                description delimited by colons and whitespace 
                                                                replaced by underscores (e.g., ":sad_face:")
    """
    for tweet in dataset:
        #remove emojis of a tweet
        tweet_without_emoji = emoji.replace_emoji(tweet["tweetWithoutURL"], replace='')
        #three different entries:
        #1) tweet without emojis, only text
        tweet["tweetWithoutEmoji"] = tweet_without_emoji
        #2) tweet with text AND description of emoji (e.g. ":sad_face:")
        tweet["tweetWithEmojiDescription"] = emoji.demojize(tweet["tweetwithoutURL"])
         #3) list of all emojis in tweet (again, description of the emoji)
        tweet["emojis"] = [emoji.demojize(entry['emoji']) for entry in emoji.emoji_list(tweet["tweetwithoutURL"])]
        # Substitute ":" by '' and "_" by " " for further processing
        for emoj in tweet["emojis"]:
            modified_emoji = re.sub(r':', '', emoj)
            modified_emoji = re.sub(r'_', ' ', modified_emoji)
            # Add <e> and <\e> tags around the modified emoji
            modified_emoji_with_tags = "<e> " + modified_emoji + " </e>"
            tweet["tweetWithEmojiDescription"] = tweet["tweetWithEmojiDescription"].replace(emoj, modified_emoji_with_tags)
    return dataset


def tokenizer(dataset: list):
    """
    Function that tokenizes the text in a given dataset.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            The dictionary key "tweetWithoutEmoji" contains the
                            text which is supposed to be tokenized.

    Output:
    1. dataset (list)   :   Inputted dataset that has been modified to include
                            the tokenized tweet, added under the new key
                            "tweetTokenized". 
    """
    # create object for tokenizer
    tokenizer = SoMaJo("en_PTB", split_sentences=False)

    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        to_tokenize = [tweet["tweetWithoutEmoji"]]  # must be list for tokenizer
        # tokenize the sentence (without emojis)
        tokenized_obj = tokenizer.tokenize_text(to_tokenize)
        for paragraph in tokenized_obj:
            tokenized_tweet = [token.text for token in paragraph]
        # create a new dict-element with the tokenized text
        tweet["tweetTokenized"] = tokenized_tweet
    return dataset


dataset = load_dataset(DATA_PATH)
dataset = cleanURLS(dataset)
dataset = handling_emojis(dataset)
dataset = tokenizer(dataset)


########### FEATURE EXTRACTION ###########


def pos_tagging(dataset):
    """
    Function that attaches POS tags to the tokenized text in the dataset.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            FURTHER INFO TO BE ADDED 

    Output:
    TO BE DETERMINED
    """
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


def sentiment(dataset):
    """
    Function that calculates the sentiment score.
    MORE INFO TO BE ADDED.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            FURTHER INFO TO BE ADDED   

    Output:
    TO BE DETERMINED
    """
    for tweet in dataset:
        scores = analyzer.polarity_scores(tweet["tweetwithEmojidescription"])
        tweet["Sentiment"] = scores
        #store range between positive and negative sentiment
        tweet["polarity"] = tweet["Sentiment"]["pos"] - tweet["Sentiment"]["neg"]
    return dataset


print(dataset)

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
