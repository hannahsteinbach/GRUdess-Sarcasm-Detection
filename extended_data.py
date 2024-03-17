import csv
import html
import random
import re
from tqdm import tqdm

DATA_PATH_OG = './data/train.En__1_.csv'
DATA_PATH_extended = './data/extended_dataset.txt'

def load_OG_dataset(filename: str):
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
                # rephrases get their own dictionary entry
                # same ID as original tweets, we still know what the rephrase/OG tweet is bc of class
                rephrase_dict = dict()
                rephrase_dict["tweetID"] = decoded_row[0]
                rephrase_dict["tweetText"] = decoded_row[3]
                rephrase_dict["class"] = 0
                dataset.append(rephrase_dict)
    return dataset


def load_extended_dataset(filename: str):
    """
    Function that reads the extended dataset, converts HTML encoded text to plain text,
    and saves information about a tweet within a dictionary. It is the same as load_dataset, but with minor changes,
    i.e., we do not have a text ID and the enumeration of tweet text and class is different, there are no rephrases
    of the sarcastic tweets.

    Input:
    1. filename (str)   :   Name of the file that is supposed to be read in.

    Output:
    1. dataset (list)   :   List of dictionaries where each element is a
                            Tweet from the given dataset. Each dictionary has
                            the following keys:
                                tweetText   :   text of the tweet formatted as a string
                                class       :   classification of the tweet as
                                                non-sarcastic (0) or sarcastic (1)
    """
    dataset = list()
    with open(filename, "r", encoding="UTF8") as f:

        data = csv.reader(f, delimiter="\t")

        # this just skips the first line
        next(data)

        for row in data:
            # this converts html encoded text to plain text (solves &amp problem)
            decoded_row = [html.unescape(cell) for cell in row]
            tweet_dict = dict()
            tweet_dict["tweetText"] = decoded_row[2]
            tweet_dict["class"] = int(decoded_row[1])
            dataset.append(tweet_dict)
    return dataset


def create_balanced_dataset(originaldataset: list, extension: list):
    """
        Function that uses the extended dataset to extract additional sarcastic tweets to add those to the original
        dataset and thus create a balanced dataset.

        Input:
        1. originaldataset  :   The original dataset that was created with the function load_OG_dataset().
        2. extension        :   The extended dataset that was created with the function load_extended_dataset().

        Output:
        1. dataset (list)   :   List of dictionaries where the original dataset is extended by more sarcastic tweets
                                from the extended dataset, so that the resulting dataset has the same number of
                                sarcastic and non-sarcastic tweets.
        """
    random.seed(4) #set a seed for consistent results
    random.shuffle(extension) # shuffle the data

    count_sarcastic = 0 #count of tweets
    sarcastic_tweets = [] #store dictionaries of sarcastic tweets

    for tweet in extension:
        if tweet["class"] == 1:
            if count_sarcastic < 2601: #keep track of added tweets
                sarcastic_tweets.append(tweet)
                count_sarcastic += 1 #to count tweets
    balanced = sarcastic_tweets + originaldataset #add additional sarcastic tweets to the original dataset
    return balanced #return the balanced dataset


def create_unbalanced_dataset(originaldataset, extension):
    """
        Function that uses the extended dataset to extract additional non-sarcastic AND
        sarcastic tweets to add those to the original dataset. This will return an extended dataset that retains
        the original sarcastic-to-non-sarcastic ratio.

        Input:
        1. originaldataset  :   The original dataset that was created with the function load_OG_dataset().
        2. extension        :   The extended dataset that was created with the function load_extended_dataset().

        Output:
        1. dataset (list)   :   List of dictionaries where the original dataset is extended by more sarcastic and
                                non-sarcastic tweets from the extended dataset, so that the resulting dataset has double
                                the number of sarcastic and non-sarcastic tweets than the original one.
        """
    random.seed(8) #seed for consistent results
    random.shuffle(extension) #shuffle the data

    count_sarcastic = 0 #keep track of added sarcastic tweets
    count_unsarcastic = 0 #keep track of added unsarcastuc tweets
    tweets = [] #all added tweets

    for tweet in extension:
        if tweet["class"] == 1:
            if count_sarcastic < 867: #number of sarcastic tweets in OG dataset
                tweets.append(tweet)
                count_sarcastic += 1
        else:
            if count_unsarcastic < 3468: #number of non-sarcastic tweets in OG dataset
                tweets.append(tweet)
                count_unsarcastic += 1
    unbalanced = tweets + originaldataset #add additional (non-)sarcastic tweets to the original dataset
    return unbalanced #return the extended, but unbalanced dataset (same ratio as OG dataset)


def cleanURLS(dataset: list):
    """
    Function that removes URLs from our analysis.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            The dictionary key "tweetText" contains the
                            text to be modified for URL removal.

    Output:
    1. dataset (list)   :   Inputted dataset that contains tweet versions
                            that are completely free of URLs. These URL-free tweets
                            are the new values for "tweetText".
    """

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Handling URLs")

    url_pattern = re.compile(r' https?://\S+|www\.\S+')
    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        tweet_toclean = tweet["tweetText"]
        # remove URLs from text
        no_URL = url_pattern.sub('', tweet_toclean)
        # create a new dict-element with the clean text
        tweet["tweetText"] = no_URL

        progressBar.update(1)

    progressBar.close()

    return dataset

def cleanRTsarcasm(dataset: list):
    """
    Function that removes sarcasm-specific hashtags as well as signs of retweeting (RT) from our analysis.
    This is necessary since the extended dataset includes these features while the original dataset does not.

    Input:
    1. dataset (list)   :   A list of dictionaries.
                            The dictionary key "tweetText" contains the
                            text to be modified for the removal of #sarcasm, #not, #sarcastic, #sarcastictweet and 'RT'.

    Output:
    1. dataset (list)   :   Inputted dataset that contains tweet versions
                            that are completely free of #sarcasm, #not, #sarcastic, #sarcastictweet and 'RT'.
                            These modified tweets are the new values for "tweetText".
    """

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Handling RT and #sarcasm")

    sarcasm_pattern = re.compile(r' #(s|S)arcasm| #(s|S)arcastic(t|T)weet| #(s|S)arcastic | #(n|N)(O|o)(t|T)h\b')
    RT_pattern = re.compile(r'^RT ')
    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        tweet_toclean = tweet["tweetText"]
        # remove 'RT' from text
        no_RT = RT_pattern.sub('', tweet_toclean)
        # remove sarcasm-specific hashtags from text
        no_sarcasm = sarcasm_pattern.sub('', no_RT)
        # create a new dict-element with the clean text
        tweet["tweetText"] = no_sarcasm

        progressBar.update(1)

    progressBar.close()

    return dataset


OG_dataset = load_OG_dataset(DATA_PATH_OG) #load dataset
OG_dataset = cleanURLS(OG_dataset) #clean the URLs

extended_dataset = load_extended_dataset(DATA_PATH_extended) #load the extended dataset

balanced_dataset = create_balanced_dataset(OG_dataset, extended_dataset) #create a balanced dataset
balanced_dataset = cleanURLS(balanced_dataset) #clean the URLs
balanced_dataset = cleanRTsarcasm(balanced_dataset) #clean from sarcasm-specific hashtags and 'RT'

unbalanced_dataset = create_unbalanced_dataset(OG_dataset, extended_dataset) #create an unbalanced dataset
unbalanced_dataset = cleanURLS(unbalanced_dataset) #clean the URLs
unbalanced_dataset = cleanRTsarcasm(unbalanced_dataset) #clean from sarcasm-specific hashtags and 'RT'
