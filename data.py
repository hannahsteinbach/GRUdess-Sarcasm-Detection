import csv
from nltk.tokenize import word_tokenize
import emoji

DATA_PATH = './data/train.SemEval.csv'

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
