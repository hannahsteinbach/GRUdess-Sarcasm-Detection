import csv
import html
import random
import re
import json

DATA_PATH_OG = './data/train.En__1_.csv'
DATA_PATH_extended = './data/extended_dataset.txt'
DATA_PATH_testset = './data/task_A_En_test__1_.csv'

class TweetDataset:
    def __init__(self):
        self.OG_dataset = []
        self.testset = []
        self.extension = []
        self.balanced_dataset = []
        self.unbalanced_dataset = []

    def load_OG_dataset(self, filename):
        """
        Function that reads the dataset, converts HTML encoded text to plain text,
        and saves information about a tweet within a dictionary. This function fills self.OG_dataset.

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

        with open(filename, "r", encoding="UTF8") as f:
            data = csv.reader(f, delimiter=",")
            next(data)  # Skip header
            for row in data:
                decoded_row = [html.unescape(cell) for cell in row]
                tweet_dict = {
                    "tweetText": decoded_row[1],
                    "class": int(decoded_row[2])
                }
                self.OG_dataset.append(tweet_dict)
                if tweet_dict["class"] == 1:
                    rephrase_dict = {
                        "tweetText": decoded_row[3],
                        "class": 0
                    }
                    self.OG_dataset.append(rephrase_dict)

    def load_extended_dataset(self, filename):
        """
        Function that reads the extended dataset, converts HTML encoded text to plain text,
        and saves information about a tweet within a dictionary. It is the same as load_dataset, but with minor changes,
        i.e., we do not have a text ID and the enumeration of tweet text and class is different, there are no rephrases
        of the sarcastic tweets. This function fills self.extension.

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
        with open(filename, "r", encoding="UTF8") as f:
            data = csv.reader(f, delimiter="\t")
            next(data)  # Skip header
            for row in data:
                decoded_row = [html.unescape(cell) for cell in row]
                tweet_dict = {
                    "tweetText": decoded_row[2],
                    "class": int(decoded_row[1])
                }
                self.extension.append(tweet_dict)

    def load_test_dataset(self, filename):
        """
        Function that reads the test dataset, converts HTML encoded text to plain text,
        and saves information about a tweet within a dictionary. It is the same as load_dataset, but with minor changes,
        i.e., we do not have a text ID and the enumeration of tweet text and class is different, there are no rephrases
        of the sarcastic tweets. This function fills self.extension.

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
        with open(filename, "r", encoding="UTF8") as f:
            data = csv.reader(f, delimiter=",")
            next(data)  # Skip header
            for row in data:
                decoded_row = [html.unescape(cell) for cell in row]
                tweet_dict = {
                    "tweetText": decoded_row[0],
                    "class": int(decoded_row[1])
                }
                self.testset.append(tweet_dict)

    def create_balanced_dataset(self):
        """
        Function that uses the extended dataset (self.extension) to extract additional sarcastic tweets to add those to
        the original dataset (self.OG_dataset) and thus create a new dataset. This function fills self.balanced_dataset
        accordingly, resulting in a balanced dataset consisting of all tweets from the original dataset and some
        of the extended dataset.
        """
        random.seed(4)  # set a seed for consistent results
        random.shuffle(self.extension)  # shuffle the data

        count_sarcastic = 0  # count of tweets
        sarcastic_tweets = []  # store dictionaries of sarcastic tweets

        for tweet in self.extension:
            if tweet["class"] == 1:
                if count_sarcastic < 2601:  # keep track of added tweets
                    sarcastic_tweets.append(tweet)
                    count_sarcastic += 1  # to count tweets
        self.balanced_dataset = self.OG_dataset + sarcastic_tweets  # add additional sarcastic tweets to the original dataset

    def create_unbalanced_dataset(self):
        """
        Function that uses the extended dataset (self.extension) to extract additional non-sarcastic AND
        sarcastic tweets to add those to the original dataset (self.OG_dataset). This fills self.unbalanced_dataset
        accordingly, resulting in a balanced dataset consisting of all tweets from the original dataset and some
        of the extended dataset.
        """
        random.seed(8)  # seed for consistent results
        random.shuffle(self.extension)  # shuffle the data

        count_sarcastic = 0  # keep track of added sarcastic tweets
        count_unsarcastic = 0  # keep track of added unsarcastuc tweets
        tweets = []  # all added tweets

        for tweet in self.extension:
            if tweet["class"] == 1:
                if count_sarcastic < 867:  # number of sarcastic tweets in OG dataset
                    tweets.append(tweet)
                    count_sarcastic += 1
            else:
                if count_unsarcastic < 3468:  # number of non-sarcastic tweets in OG dataset
                    tweets.append(tweet)
                    count_unsarcastic += 1
        self.unbalanced_dataset = self.OG_dataset + tweets  # add additional (non-)sarcastic tweets to the original dataset

    def clean_urls(self, dataset):
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

        url_pattern = re.compile(r'( |)https?://\S+|www\.\S+')
        for tweet in dataset:
            tweet["tweetText"] = url_pattern.sub('', tweet["tweetText"])

    def clean_rts_sarcasm(self, dataset):
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
        sarcasm_pattern = re.compile(r'( |)#(s|S)arcasm|( |)#(s|S)arcastic(t|T)weet|( |)#(s|S)arcastic|( |)#(n|N)(O|o)(t|T)h\b')
        RT_pattern = re.compile(r'^RT ')
        for tweet in dataset:
            tweet["tweetText"] = RT_pattern.sub('', tweet["tweetText"])
            tweet["tweetText"] = sarcasm_pattern.sub('', tweet["tweetText"])

    # convert dataset to jsonlines for POS tagging
    def dataset_to_jsonlines(self, dataset, outputFile):
        with open(outputFile, 'w', encoding='utf-8') as jsonlinesfile:
            for tweet in dataset:
                data = {'text': tweet["tweetText"]}
                jsonline = json.dumps(data, ensure_ascii=False) + '\n'
                jsonlinesfile.write(jsonline)


# Create an instance of the TweetDataset class
dataset = TweetDataset()

# Load the original and extended datasets for training as well as the testset
dataset.load_test_dataset(DATA_PATH_testset)
dataset.load_OG_dataset(DATA_PATH_OG)
dataset.load_extended_dataset(DATA_PATH_extended)

# Create the balanced and unbalanced datasets
dataset.create_unbalanced_dataset()
dataset.create_balanced_dataset()

# Clean URLs
dataset.clean_urls(dataset.balanced_dataset)
dataset.clean_urls(dataset.unbalanced_dataset)
dataset.clean_urls(dataset.OG_dataset)
dataset.clean_urls(dataset.testset)

# Clean RTs and sarcasm-related patterns
dataset.clean_rts_sarcasm(dataset.balanced_dataset)
dataset.clean_rts_sarcasm(dataset.unbalanced_dataset)

# Assign the original dataset to a separate variable to later pass it to preprocessing.py
OG_dataset = dataset.OG_dataset
balanced_dataset = dataset.balanced_dataset
unbalanced_dataset = dataset.unbalanced_dataset
testset = dataset.testset

# slightly modified dict of emoticons by
# https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
# Modification: removed "or smiley" terms
# added :p, :s, ¯\\_(ツ)_/¯, xP
emoticons_dic= {
            u"¯\\_(ツ)_/¯": "shrug of the shoulders",
            u":‑)": "Happy face",
            u":-))": "Very Happy face",
            u":-)))": "Very very Happy face",
            u":)": "Happy face",
            u":))": "Very Happy face",
            u":)))": "Very very Happy face",
            u":-]": "Happy face",
            u":]": "Happy face",
            u":-3": "Happy face",
            u":3": "Happy face",
            u":->": "Happy face",
            u":>": "Happy face",
            u"8-)": "Happy face",
            u":o)": "Happy face",
            u":-}": "Happy face",
            u":}": "Happy face",
            u":-)": "Happy face",
            u":c)": "Happy face",
            u":^)": "Happy face",
            u"=]": "Happy face",
            u"=)": "Happy face",
            u":‑D": "Laughing, big grin or laugh with glasses",
            u":D": "Laughing, big grin or laugh with glasses",
            u"8‑D": "Laughing, big grin or laugh with glasses",
            u"8D": "Laughing, big grin or laugh with glasses",
            u"X‑D": "Laughing, big grin or laugh with glasses",
            u"XD": "Laughing, big grin or laugh with glasses",
            u"xD": "Laughing, big grin or laugh with glasses",
            u"=D": "Laughing, big grin or laugh with glasses",
            u"=3": "Laughing, big grin or laugh with glasses",
            u"B^D": "Laughing, big grin or laugh with glasses",
            u":-))": "Very happy",
            u":-(": "Frown, sad, angry or pouting",
            u":‑(": "Frown, sad, angry or pouting",
            u":(": "Frown, sad, angry or pouting",
            u":‑c": "Frown, sad, angry or pouting",
            u":c": "Frown, sad, angry or pouting",
            u":‑<": "Frown, sad, angry or pouting",
            u":<": "Frown, sad, angry or pouting",
            u":‑[": "Frown, sad, angry or pouting",
            u":[": "Frown, sad, angry or pouting",
            u":-||": "Frown, sad, angry or pouting",
            u">:[": "Frown, sad, angry or pouting",
            u":{": "Frown, sad, angry or pouting",
            u":@": "Frown, sad, angry or pouting",
            u">:(": "Frown, sad, angry or pouting",
            u":'‑(": "Crying",
            u":'(": "Crying",
            u":'‑)": "Tears of happiness",
            u":')": "Tears of happiness",
            u"D‑':": "Horror",
            u"D:<": "Disgust",
            u"D:": "Sadness",
            u"D8": "Great dismay",
            u"D;": "Great dismay",
            u"D=": "Great dismay",
            u"DX": "Great dismay",
            u":‑O": "Surprise",
            u":O": "Surprise",
            u":‑o": "Surprise",
            u":o": "Surprise",
            u":-0": "Shock",
            u"8‑0": "Yawn",
            u">:O": "Yawn",
            u":-*": "Kiss",
            u":*": "Kiss",
            u":X": "Kiss",
            u";‑)": "Wink or smirk",
            u";)": "Wink or smirk",
            u"*-)": "Wink or smirk",
            u"*)": "Wink or smirk",
            u";‑]": "Wink or smirk",
            u";]": "Wink or smirk",
            u";^)": "Wink or smirk",
            u":‑,": "Wink or smirk",
            u";D": "Wink or smirk",
            u":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"xP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u">:[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=[(\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u"=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":s": "Skeptical, annoyed, undecided, uneasy or hesitant",
            u":‑|": "Straight face",
            u":|": "Straight face",
            u":$": "Embarrassed or blushing",
            u":‑x": "Sealed lips or wearing braces or tongue-tied",
            u":x": "Sealed lips or wearing braces or tongue-tied",
            u":‑#": "Sealed lips or wearing braces or tongue-tied",
            u":#": "Sealed lips or wearing braces or tongue-tied",
            u":‑&": "Sealed lips or wearing braces or tongue-tied",
            u":&": "Sealed lips or wearing braces or tongue-tied",
            u"O:‑)": "Angel, saint or innocent",
            u"O:)": "Angel, saint or innocent",
            u"0:‑3": "Angel, saint or innocent",
            u"0:3": "Angel, saint or innocent",
            u"0:‑)": "Angel, saint or innocent",
            u"0:)": "Angel, saint or innocent",
            u":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
            u"0;^)": "Angel, saint or innocent",
            u">:‑)": "Evil or devilish",
            u">:)": "Evil or devilish",
            u"}:‑)": "Evil or devilish",
            u"}:)": "Evil or devilish",
            u"3:‑)": "Evil or devilish",
            u"3:)": "Evil or devilish",
            u">;)": "Evil or devilish",
            u"|;‑)": "Cool",
            u"|‑O": "Bored",
            u":‑J": "Tongue-in-cheek",
            u"#‑)": "Party all night",
            u"%‑)": "Drunk or confused",
            u"%)": "Drunk or confused",
            u":-###..": "Being sick",
            u":###..": "Being sick",
            u"<:‑|": "Dump",
            u"(>_<)": "Troubled",
            u"(>_<)>": "Troubled",
            u"(';')": "Baby",
            u"(^^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"(^_^;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"(-_-;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"(~_~;) (・.・;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
            u"(-_-)zzz": "Sleeping",
            u"(^_-)": "Wink",
            u"((+_+))": "Confused",
            u"(+o+)": "Confused",
            u"(o|o)": "Ultraman",
            u"^_^": "Joyful",
            u"(^_^)/": "Joyful",
            u"(^O^)／": "Joyful",
            u"(^o^)／": "Joyful",
            u"(__)": "Kowtow as a sign of respect, or dogeza for apology",
            u"_(._.)_": "Kowtow as a sign of respect, or dogeza for apology",
            u"<(_ _)>": "Kowtow as a sign of respect, or dogeza for apology",
            u"<m(__)m>": "Kowtow as a sign of respect, or dogeza for apology",
            u"m(__)m": "Kowtow as a sign of respect, or dogeza for apology",
            u"m(_ _)m": "Kowtow as a sign of respect, or dogeza for apology",
            u"('_')": "Sad or Crying",
            u"(/_;)": "Sad or Crying",
            u"(T_T) (;_;)": "Sad or Crying",
            u"(;_;": "Sad of Crying",
            u"(;_:)": "Sad or Crying",
            u"(;O;)": "Sad or Crying",
            u"(:_;)": "Sad or Crying",
            u"(ToT)": "Sad or Crying",
            u";_;": "Sad or Crying",
            u";-;": "Sad or Crying",
            u";n;": "Sad or Crying",
            u";;": "Sad or Crying",
            u"Q.Q": "Sad or Crying",
            u"T.T": "Sad or Crying",
            u"QQ": "Sad or Crying",
            u"Q_Q": "Sad or Crying",
            u"(-.-)": "Shame",
            u"(-_-)": "Shame",
            u"(一一)": "Shame",
            u"(；一_一)": "Shame",
            u"(=_=)": "Tired",
            u"(=^·^=)": "cat",
            u"(=^··^=)": "cat",
            u"=_^= ": "cat",
            u"(..)": "Looking down",
            u"(._.)": "Looking down",
            u"^m^": "Giggling with hand covering mouth",
            u"(・・?": "Confusion",
            u"(?_?)": "Confusion",
            u">^_^<": "Normal Laugh",
            u"<^!^>": "Normal Laugh",
            u"^/^": "Normal Laugh",
            u"（*^_^*）": "Normal Laugh",
            u"(^<^) (^.^)": "Normal Laugh",
            u"(^^)": "Normal Laugh",
            u"(^.^)": "Normal Laugh",
            u"(^_^.)": "Normal Laugh",
            u"(^_^)": "Normal Laugh",
            u"(^^)": "Normal Laugh",
            u"(^J^)": "Normal Laugh",
            u"(*^.^*)": "Normal Laugh",
            u"(^—^）": "Normal Laugh",
            u"(#^.^#)": "Normal Laugh",
            u"（^—^）": "Waving",
            u"(;_;)/~~~": "Waving",
            u"(^.^)/~~~": "Waving",
            u"(-_-)/~~~ ($··)/~~~": "Waving",
            u"(T_T)/~~~": "Waving",
            u"(ToT)/~~~": "Waving",
            u"(*^0^*)": "Excited",
            u"(*_*)": "Amazed",
            u"(*_*;": "Amazed",
            u"(+_+) (@_@)": "Amazed",
            u"(*^^)v": "Laughing,Cheerful",
            u"(^_^)v": "Laughing,Cheerful",
            u"((d[-_-]b))": "Headphones,Listening to music",
            u'(-"-)': "Worried",
            u"(ーー;)": "Worried",
            u"(^0_0^)": "Eyeglasses",
            u"(＾ｖ＾)": "Happy",
            u"(＾ｕ＾)": "Happy",
            u"(^)o(^)": "Happy",
            u"(^O^)": "Happy",
            u"(^o^)": "Happy",
            u")^o^(": "Happy",
            u":O o_O": "Surprised",
            u"o_0": "Surprised",
            u"o.O": "Surpised",
            u"(o.o)": "Surprised",
            u"oO": "Surprised",
            u"(*￣m￣)": "Dissatisfied",
            u"(‘A`)": "Snubbed or Deflated",
        }

