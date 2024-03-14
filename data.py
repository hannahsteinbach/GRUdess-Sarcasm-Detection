from tqdm import tqdm
import csv
#from nltk.tokenize import word_tokenize
from somajo import SoMaJo
from someweta import ASPTagger
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import emoji
import html
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

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
                            "tweetWithoutURL" has been added
                            accordingly.
    """

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Handling URLs")

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        tweet_toclean = tweet["tweetText"]
        # remove URLs from text
        no_URL = url_pattern.sub('', tweet_toclean)
        # create a new dict-element with the clean text
        tweet["tweetWithoutURL"] = no_URL

        progressBar.update(1)

    progressBar.close()

    return dataset


def handling_emojis(dataset: list):
    """
    Function that separates and prepares emojis for further processing.

    Input:
    1. dataset (list)   :   A list of dictionaries. The dictionary key "tweetWithoutURL" contains the
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

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Handling Emojis")

    for tweet in dataset:
        #remove emojis of a tweet
        tweet_without_emoji = emoji.replace_emoji(tweet["tweetWithoutURL"], replace='')
        #three different entries:
        #1) tweet without emojis, only text
        tweet["tweetWithoutEmoji"] = tweet_without_emoji
        #2) tweet with text AND description of emoji (e.g. ":sad_face:")
        tweet["tweetWithEmojiDescription"] = emoji.demojize(tweet["tweetWithoutURL"])
         #3) list of all emojis in tweet (again, description of the emoji)
        tweet["emojis"] = [emoji.demojize(entry['emoji']) for entry in emoji.emoji_list(tweet["tweetWithoutURL"])]
        # Substitute ":" by '' and "_" by " " for further processing
        for emoj in tweet["emojis"]:
            modified_emoji = re.sub(r':', '', emoj)
            modified_emoji = re.sub(r'_', ' ', modified_emoji)
            # Add <e> and <\e> tags around the modified emoji
            modified_emoji_with_tags = "<e> " + modified_emoji + " </e>"
            tweet["tweetWithEmojiDescription"] = tweet["tweetWithEmojiDescription"].replace(emoj, modified_emoji_with_tags)

        progressBar.update(1)

    progressBar.close()

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

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Tokenizing")

    # create object for tokenizer
    tokenizer = SoMaJo("en_PTB", split_sentences=False)

    # go through the dataset
    for tweet in dataset:
        # take the text element from the dict
        to_tokenize = [tweet["tweetText"]]  # must be list for tokenizer
        # tokenize the sentence (without emojis)
        tokenized_obj = tokenizer.tokenize_text(to_tokenize)
        for paragraph in tokenized_obj:
            tokenized_tweet = [token.text for token in paragraph]
        # create a new dict-element with the tokenized text
        tweet["tweetTokenized"] = tokenized_tweet

        progressBar.update(1)

    progressBar.close()
    
    return dataset


def handling_emoticons(dataset: list):
    """
    
    """
    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Handling Emoticons")

    # slightly modified dict of emoticons by 
    # https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
    # Modification: removed "or smiley" terms
    # added :p, :s, ¯\\_(ツ)_/¯

    # dictionary of emoticons
    EMOTICONS_EMO = {
        u"¯\\_(ツ)_/¯": "shrug of the shoulders",
        u":‑)":"Happy face",
        u":-))":"Very Happy face",
        u":-)))":"Very very Happy face",
        u":)":"Happy face",
        u":))":"Very Happy face",
        u":)))":"Very very Happy face",
        u":-]":"Happy face",
        u":]":"Happy face",
        u":-3":"Happy face",
        u":3":"Happy face",
        u":->":"Happy face",
        u":>":"Happy face",
        u"8-)":"Happy face",
        u":o)":"Happy face",
        u":-}":"Happy face",
        u":}":"Happy face",
        u":-)":"Happy face",
        u":c)":"Happy face",
        u":^)":"Happy face",
        u"=]":"Happy face",
        u"=)":"Happy face",
        u":‑D":"Laughing, big grin or laugh with glasses",
        u":D":"Laughing, big grin or laugh with glasses",
        u"8‑D":"Laughing, big grin or laugh with glasses",
        u"8D":"Laughing, big grin or laugh with glasses",
        u"X‑D":"Laughing, big grin or laugh with glasses",
        u"XD":"Laughing, big grin or laugh with glasses",
        u"xD":"Laughing, big grin or laugh with glasses",
        u"=D":"Laughing, big grin or laugh with glasses",
        u"=3":"Laughing, big grin or laugh with glasses",
        u"B^D":"Laughing, big grin or laugh with glasses",
        u":-))":"Very happy",
        u":-(":"Frown, sad, angry or pouting",
        u":‑(":"Frown, sad, angry or pouting",
        u":(":"Frown, sad, angry or pouting",
        u":‑c":"Frown, sad, angry or pouting",
        u":c":"Frown, sad, angry or pouting",
        u":‑<":"Frown, sad, angry or pouting",
        u":<":"Frown, sad, angry or pouting",
        u":‑[":"Frown, sad, angry or pouting",
        u":[":"Frown, sad, angry or pouting",
        u":-||":"Frown, sad, angry or pouting",
        u">:[":"Frown, sad, angry or pouting",
        u":{":"Frown, sad, angry or pouting",
        u":@":"Frown, sad, angry or pouting",
        u">:(":"Frown, sad, angry or pouting",
        u":'‑(":"Crying",
        u":'(":"Crying",
        u":'‑)":"Tears of happiness",
        u":')":"Tears of happiness",
        u"D‑':":"Horror",
        u"D:<":"Disgust",
        u"D:":"Sadness",
        u"D8":"Great dismay",
        u"D;":"Great dismay",
        u"D=":"Great dismay",
        u"DX":"Great dismay",
        u":‑O":"Surprise",
        u":O":"Surprise",
        u":‑o":"Surprise",
        u":o":"Surprise",
        u":-0":"Shock",
        u"8‑0":"Yawn",
        u">:O":"Yawn",
        u":-*":"Kiss",
        u":*":"Kiss",
        u":X":"Kiss",
        u";‑)":"Wink or smirk",
        u";)":"Wink or smirk",
        u"*-)":"Wink or smirk",
        u"*)":"Wink or smirk",
        u";‑]":"Wink or smirk",
        u";]":"Wink or smirk",
        u";^)":"Wink or smirk",
        u":‑,":"Wink or smirk",
        u";D":"Wink or smirk",
        u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=[(\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":s":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":‑|":"Straight face",
        u":|":"Straight face",
        u":$":"Embarrassed or blushing",
        u":‑x":"Sealed lips or wearing braces or tongue-tied",
        u":x":"Sealed lips or wearing braces or tongue-tied",
        u":‑#":"Sealed lips or wearing braces or tongue-tied",
        u":#":"Sealed lips or wearing braces or tongue-tied",
        u":‑&":"Sealed lips or wearing braces or tongue-tied",
        u":&":"Sealed lips or wearing braces or tongue-tied",
        u"O:‑)":"Angel, saint or innocent",
        u"O:)":"Angel, saint or innocent",
        u"0:‑3":"Angel, saint or innocent",
        u"0:3":"Angel, saint or innocent",
        u"0:‑)":"Angel, saint or innocent",
        u"0:)":"Angel, saint or innocent",
        u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"0;^)":"Angel, saint or innocent",
        u">:‑)":"Evil or devilish",
        u">:)":"Evil or devilish",
        u"}:‑)":"Evil or devilish",
        u"}:)":"Evil or devilish",
        u"3:‑)":"Evil or devilish",
        u"3:)":"Evil or devilish",
        u">;)":"Evil or devilish",
        u"|;‑)":"Cool",
        u"|‑O":"Bored",
        u":‑J":"Tongue-in-cheek",
        u"#‑)":"Party all night",
        u"%‑)":"Drunk or confused",
        u"%)":"Drunk or confused",
        u":-###..":"Being sick",
        u":###..":"Being sick",
        u"<:‑|":"Dump",
        u"(>_<)":"Troubled",
        u"(>_<)>":"Troubled",
        u"(';')":"Baby",
        u"(^^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"(^_^;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"(-_-;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"(~_~;) (・.・;)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"(-_-)zzz":"Sleeping",
        u"(^_-)":"Wink",
        u"((+_+))":"Confused",
        u"(+o+)":"Confused",
        u"(o|o)":"Ultraman",
        u"^_^":"Joyful",
        u"(^_^)/":"Joyful",
        u"(^O^)／":"Joyful",
        u"(^o^)／":"Joyful",
        u"(__)":"Kowtow as a sign of respect, or dogeza for apology",
        u"_(._.)_":"Kowtow as a sign of respect, or dogeza for apology",
        u"<(_ _)>":"Kowtow as a sign of respect, or dogeza for apology",
        u"<m(__)m>":"Kowtow as a sign of respect, or dogeza for apology",
        u"m(__)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"m(_ _)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"('_')":"Sad or Crying",
        u"(/_;)":"Sad or Crying",
        u"(T_T) (;_;)":"Sad or Crying",
        u"(;_;":"Sad of Crying",
        u"(;_:)":"Sad or Crying",
        u"(;O;)":"Sad or Crying",
        u"(:_;)":"Sad or Crying",
        u"(ToT)":"Sad or Crying",
        u";_;":"Sad or Crying",
        u";-;":"Sad or Crying",
        u";n;":"Sad or Crying",
        u";;":"Sad or Crying",
        u"Q.Q":"Sad or Crying",
        u"T.T":"Sad or Crying",
        u"QQ":"Sad or Crying",
        u"Q_Q":"Sad or Crying",
        u"(-.-)":"Shame",
        u"(-_-)":"Shame",
        u"(一一)":"Shame",
        u"(；一_一)":"Shame",
        u"(=_=)":"Tired",
        u"(=^·^=)":"cat",
        u"(=^··^=)":"cat",
        u"=_^= ":"cat",
        u"(..)":"Looking down",
        u"(._.)":"Looking down",
        u"^m^":"Giggling with hand covering mouth",
        u"(・・?":"Confusion",
        u"(?_?)":"Confusion",
        u">^_^<":"Normal Laugh",
        u"<^!^>":"Normal Laugh",
        u"^/^":"Normal Laugh",
        u"（*^_^*）" :"Normal Laugh",
        u"(^<^) (^.^)":"Normal Laugh",
        u"(^^)":"Normal Laugh",
        u"(^.^)":"Normal Laugh",
        u"(^_^.)":"Normal Laugh",
        u"(^_^)":"Normal Laugh",
        u"(^^)":"Normal Laugh",
        u"(^J^)":"Normal Laugh",
        u"(*^.^*)":"Normal Laugh",
        u"(^—^）":"Normal Laugh",
        u"(#^.^#)":"Normal Laugh",
        u"（^—^）":"Waving",
        u"(;_;)/~~~":"Waving",
        u"(^.^)/~~~":"Waving",
        u"(-_-)/~~~ ($··)/~~~":"Waving",
        u"(T_T)/~~~":"Waving",
        u"(ToT)/~~~":"Waving",
        u"(*^0^*)":"Excited",
        u"(*_*)":"Amazed",
        u"(*_*;":"Amazed",
        u"(+_+) (@_@)":"Amazed",
        u"(*^^)v":"Laughing,Cheerful",
        u"(^_^)v":"Laughing,Cheerful",
        u"((d[-_-]b))":"Headphones,Listening to music",
        u'(-"-)':"Worried",
        u"(ーー;)":"Worried",
        u"(^0_0^)":"Eyeglasses",
        u"(＾ｖ＾)":"Happy",
        u"(＾ｕ＾)":"Happy",
        u"(^)o(^)":"Happy",
        u"(^O^)":"Happy",
        u"(^o^)":"Happy",
        u")^o^(":"Happy",
        u":O o_O":"Surprised",
        u"o_0":"Surprised",
        u"o.O":"Surpised",
        u"(o.o)":"Surprised",
        u"oO":"Surprised",
        u"(*￣m￣)":"Dissatisfied",
        u"(‘A`)":"Snubbed or Deflated",
    }
    
    # go through each tweet 
    for tweet in dataset:
        # check for each token if it consists of only alphabetical 
        # characters
        for token in tweet["tweetTokenized"]:
            check_emoticon = token.isalpha()

            if check_emoticon == False and len(token) >= 2:
                # if False, check which one it is

                """
                    Comment regarding inefficiency:
                    This could have been also handled the other way around,
                    look, if the token is within the keys BUT
                    there could be cases with :DDD, where the emoticon
                    is exaggerated. These cases would become lost
                """
                for key in EMOTICONS_EMO.keys():
                    if key in token:

                        # append emoticon to list of emojis
                        tweet["emojis"].append(EMOTICONS_EMO[key])

                        # create replacement
                        replacement = "<e> " + EMOTICONS_EMO[key] + " </e>"
                        
                        # replace emoticon with the description
                        tweet["tweetWithEmojiDescription"] = tweet["tweetWithEmojiDescription"].replace(token, replacement)

                        # delete emoticon from other entry
                        tweet["tweetWithoutEmoji"] = tweet["tweetWithoutEmoji"].replace(token, "")

                        break

            elif token == "xD" or token == "XD":
                replacement = "<e> " + EMOTICONS_EMO["xD"] + " </e>"

                # replace emoticon with the description
                tweet["tweetWithEmojiDescription"] = tweet["tweetWithEmojiDescription"].replace(token, replacement)

                # delete emoticon from other entry
                tweet["tweetWithoutEmoji"] = tweet["tweetWithoutEmoji"].replace(token, "")


            elif token == "XP":
                replacement = "<e> " + EMOTICONS_EMO["xP"] + " </e>"

                # replace emoticon with the description
                tweet["tweetWithEmojiDescription"] = tweet["tweetWithEmojiDescription"].replace(token, replacement)

                # delete emoticon from other entry
                tweet["tweetWithoutEmoji"] = tweet["tweetWithoutEmoji"].replace(token, "")

        progressBar.update(1)

    progressBar.close()

    return dataset 


def include_word2vec(dataset):
    """
        Function to include word2vec word embeddings.

        
        INFORMATION
        We could try out different models, the higher thhe
        dimension though, the more accurate the results 
    """

    total_iterations = len(dataset)
    progressBar = tqdm(total=total_iterations, desc="Word Embeddings")

    # load pre-trained model
    wv = api.load("glove-twitter-100")
    # save the vectors
    # wv.save("./used_models/vectors.kv")
    # wv = "./used_models/vectors.kv"
    for tweet in dataset:
        tokenized_sentence = tweet["tweetTokenized"]

        # create dictionary key for embeddings
        tweet["tweetTokenEmbeddings"] = list()

        # get dimension of vctor (based on chosen model)
        vector_size = wv.vector_size
        # initialize the vector
        wv_res = np.zeros(vector_size)

        # count the words that are represented as
        # vector --> later for the average of thee sentence
        counter = 0

        for token in tokenized_sentence:
            if token in wv:
                tweet["tweetTokenEmbeddings"].append(wv[token])
                # counter += 1
                # wv_res += wv[token]
                # tweet["tweetTokenEmbeddings"].append(wv_res)
            else:
                tweet["tweetTokenEmbeddings"].append(None)

        # compute average of the sentence
        # wv_res = wv_res/counter

        progressBar.update(1)

    progressBar.close()

    return dataset


dataset = load_dataset(DATA_PATH)
dataset = cleanURLS(dataset)
dataset = handling_emojis(dataset)
dataset = tokenizer(dataset)
dataset = handling_emoticons(dataset)
dataset = include_word2vec(dataset)


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
        scores = analyzer.polarity_scores(tweet["tweetWithEmojidescription"])
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
