from tqdm import tqdm
import csv
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
import numpy as np
from extended_data import emoticons_dic, OG_dataset, balanced_dataset, unbalanced_dataset, testset
import pandas as pd

class Preprocessing:
    def __init__(self, dataset):
        self.analyzer = SentimentIntensityAnalyzer()
        self.vocabulary = set()
        self.dataset = dataset
        self.bigrams = set()
        self.embeddings = []
        self.feature_array = None
        self.class_array = None


    def tokenizer(self):
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

        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Tokenizing...")

        # create object for tokenizer
        tokenizer = SoMaJo("en_PTB", split_sentences=False)

        # go through the dataset
        for tweet in self.dataset:
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

    def handling_emojis(self):
        """
        Function that separates and prepares emojis for further processing.

        Input:
        1. dataset (list)   :   A list of dictionaries. The dictionary key "tweetWithoutURL" contains the
                                text to be modified for emoji processing.

        Output:
        1. dataset (list)   :   Inputted dataset with several added dictionary entries:
                                "tweetWithoutEmojicon"         :   tweet formatted as a string without emojis/emoticons.
                                "tweetWithEmojiconDescription" :   tweet formatted as a string with emoji description;
                                                                   emojis are replaced by their description and delimited
                                                                   by "<e>" and "</e." (e.g., <e> sad face </e>)
                                "emojis"                    :   list of all emojis that appear in the tweet (e.g.😢)
                                "emojis_description"        :   list of all emojis that appear in the tweet;
                                                                each emoji is formatted as a string with emoji
                                                                description delimited by colons and whitespace
                                                                replaced by underscores (e.g., ":sad_face:")
                                "frequency_emojis"           :   stores the frequency of emojis per tweet (normalised
                                                                 by number of tokens in tweet)
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Handling Emojis...")

        for tweet in self.dataset:
            tweet["tweetWithoutEmojicon"] = emoji.replace_emoji(tweet["tweetText"], replace='')
            tweet["tweetWithEmojiconDescription"] = emoji.demojize(tweet["tweetText"])
            tweet["emojis_description"] = [emoji.demojize(entry['emoji']) for entry in emoji.emoji_list(tweet["tweetText"])]
            tweet["emojis"] = [entry['emoji'] for entry in emoji.emoji_list(tweet["tweetText"])]

            for emoj in tweet["emojis_description"]:
                modified_emoji = re.sub(r':', '', emoj)
                modified_emoji = re.sub(r'_', ' ', modified_emoji)
                modified_emoji_with_tags = "<e> " + modified_emoji + " </e>"
                tweet["tweetWithEmojiconDescription"] = tweet["tweetWithEmojiconDescription"].replace(emoj, modified_emoji_with_tags)
            if len(tweet["tweetTokenized"]) != 0:
                tweet["frequency_emojis"] = len(tweet["emojis"]) / len(tweet["tweetTokenized"])
            else:
                tweet["frequency_emojis"] = 0
            progressBar.update(1)

        progressBar.close()


    def handling_emoticons(self):
        """
        Function that separates and prepares emojis for further processing.

        Input:
        1. dataset (list)   :   A list of dictionaries. The dictionary key "tweetWithoutURL" contains the
                                text to be modified for emoticon processing.

        Output:
        1. dataset (list)   :   Inputted dataset with several added dictionary entries:
                                "tweetWithoutEmojicon"         :   tweet formatted as a string without emojis + emoticons
                                "tweetWithEmojiconDescription" :   tweet formatted as a string with emoticon description;
                                                                   emojis are replaced by their description and delimited
                                                                   by "<ec>" and "</ec." (e.g., <ec> sad face </ec>)
                                "emoticons"                    :   list of all emojis that appear in the tweet (e.g.:()
                                "emojis_description"        :   list of all emojis that appear in the tweet;
                                                                each emoji is formatted as a string with emoticon
                                                                description (from emoticon dictionary in extended_data)
                                "frequency_emoticons"           :   stores the frequency of emoticons per tweet (normalised
                                                                by number of tokens in tweet)
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Handling Emoticons...")

        # go through each tweet
        for tweet in self.dataset:
            tweet["emoticons"] = []
            tweet["emoticons_description"] = []
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
                    for key in emoticons_dic.keys():
                        if key in token:
                            # append emoticon to list of emojis
                            tweet["emoticons"].append(key)
                            tweet["emoticons_description"].append(emoticons_dic[key])

                            # create replacement
                            replacement = "<ec> " + emoticons_dic[key] + " </ec>"

                            # replace emoticon with the description
                            tweet["tweetWithEmojiconDescription"] = tweet["tweetWithEmojiconDescription"].replace(token,
                                                                                                            replacement)

                            # delete emoticon from other entry
                            tweet["tweetWithoutEmojicon"] = tweet["tweetWithoutEmojicon"].replace(token, "")

                            break

                elif token == "xD" or token == "XD":
                    replacement = "<ec> " + emoticons_dic["xD"] + " </ec>"

                    # replace emoticon with the description
                    tweet["tweetWithEmojiconDescription"] = tweet["tweetWithEmojiconDescription"].replace(token, replacement)

                    # delete emoticon from other entry
                    tweet["tweetWithoutEmojicon"] = tweet["tweetWithoutEmojicon"].replace(token, "")


                elif token == "XP":
                    replacement = "<ec> " + emoticons_dic["xP"] + " </ec>"

                    # replace emoticon with the description
                    tweet["tweetWithEmojiconDescription"] = tweet["tweetWithEmojiconDescription"].replace(token, replacement)

                    # delete emoticon from other entry
                    tweet["tweetWithoutEmojicon"] = tweet["tweetWithoutEmojicon"].replace(token, "")
            if len(tweet["tweetTokenized"]) != 0:
                tweet["frequency_emoticons"] = len(tweet["emoticons"]) / len(tweet["tweetTokenized"])
            else:
                tweet["frequency_emoticons"] = 0
            progressBar.update(1)

        progressBar.close()
    ########### PRE-PROCESSING FOR GRU ###########

    def create_vocabulary(self):
        """
            Function which creates the vocabulary for our data

            Output:
            Vocabulary attribute (self.vocabulary) for an object of the class

            INFORMATION:
                DISCUSS WHICH ONE - WITH EMOJIS OR WITH DESCRIPTIONS? IN GENERAL FOR
                ALL THE PREPROCESSING STEPS
                I TEND TO WITH DESCRIPTIONS
        """

        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Building Vocabulary...")

        for tweet in self.dataset:
            for token in tweet["tweetTokenized"]:
                self.vocabulary.add(token)

            progressBar.update(1)
        progressBar.close()

    def pos_tagging(self):
        """
        Function that attaches POS tags to the tokenized text in the dataset.

        Input:
        1. dataset (list)   :   A list of dictionaries.
                                FURTHER INFO TO BE ADDED

        Output:
        TO BE DETERMINED
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="POS-Tagging...")

        # load model
        model = "./data/english_newspaper_2017-09-15.model"

        """
        # example sentences
        sentences = [["Ein", "Satz", "ist", "eine", "Liste", "von", "Tokens", "."],
                 ["Zeitfliegen", "mögen", "einen", "Pfeil", "."]]
        """
        # create object for pos tagger
        asptagger = ASPTagger()
        asptagger.load(model)

        # go through dataset to get tokenized text
        for tweet in self.dataset:
            tokenized_tweet = [tweet["tweetTokenized"]]

            # take the sentence & pos tag it
            for sentence in tokenized_tweet:
                tagged_sentence = asptagger.tag_sentence(sentence)
                posTagged_tweet = tagged_sentence

                # create new dictionary key
                tweet["posTagged_tweet"] = posTagged_tweet #change this
            progressBar.update(1)

        progressBar.close()

    def sentiment(self):
        """
        Method that calculates the sentiment score.
        MORE INFO TO BE ADDED.

        Input:
        1. dataset (list)   :   A list of dictionaries.
                                FURTHER INFO TO BE ADDED

        Output:
        2. dictionary entry :    tweet["polarity"] contains the absolute distance between positive and negative
                                 sentiment
        TO BE DETERMINED
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Checking polarity...")
        for tweet in self.dataset:
            scores = self.analyzer.polarity_scores(tweet["tweetWithEmojiconDescription"])
            # store range between positive and negative sentiment
            tweet["polarity"] = abs(abs(scores["pos"]) - abs(scores["neg"]))
            progressBar.update(1)

        progressBar.close()

    def uni_and_bigrams(self):
        """
        Function to create unigrams and bigrams.

        Input:
        1. dataset (list)  :   A list of dictionaries.
                                The dictionary key "tweetTokenized" contains the
                                information need to create unigram and bigram sets.

        Output:
        1. dataset (list)  :   Modified dataset with unigrams and bigrams indicators (including this in the dataset for now).
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Creating unigrams and bigrams...")

        # set of all unigrams is equivalent to vocabulary
        unigrams = self.vocabulary

        # create set of all bigrams
        for tweet in self.dataset:
            tokenizedTweet = tweet["tweetTokenized"]

            # bigrams are consecutive tokens separated by a whitespace
            bigram = [tokenizedTweet[i] + " " + tokenizedTweet[i + 1] for i in range(len(tokenizedTweet) - 1)]
            self.bigrams.update(bigram)

        # we use binary indicators for unigrams and bigrams (to check whether they are present in a tweet or not)
        for tweet in self.dataset:
            tokenizedTweet = tweet["tweetTokenized"]
            bigram = [tokenizedTweet[i] + " " + tokenizedTweet[i + 1] for i in range(len(tokenizedTweet) - 1)]

            # create one-hot encoded unigram vector for each tweet
            unigramsIndicators = []

            for unigram in unigrams:
                if unigram in tokenizedTweet:
                    unigramsIndicators.append(1)
                else:
                    unigramsIndicators.append(0)

            # update unigram vector for each tweet
            tweet["unigrams "] = unigramsIndicators

            # create one-hot encoded bigram vector for each tweet
            bigramsIndicators = []

            for bigram in self.bigrams:
                if bigram in tweet["tweetText"]:
                    bigramsIndicators.append(1)
                else:
                    bigramsIndicators.append(0)

            # update bigram vector for each tweet
            tweet["bigrams"] = bigramsIndicators
            progressBar.update(1)

        progressBar.close()

    def tweet_length_tokenized(self):
        """
        Function to add tweet length as a feature.

        Input:
        1. dataset (list)   :   A list of dictionaries.
                                The dictionary key "tweetTokenized" contains the
                                text to be "measured".

        Output:
        1. dataset (list)   :   The input dataset with an additional (one item)
                                feature vector representing tweet length
                                concatenated at the end.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting lengths of tweets...")
        for tweet in self.dataset:
            tweet["tweetTokenizedLength"] = len(tweet["tweetTokenized"])
            progressBar.update(1)

        progressBar.close()

    def caps_counts(self):
        """
        Function that counts how many words in a tweet are written in all caps, how many
        have initial caps and how many capitalized letters there are overall in a tweet.

        Input:
        1. dataset (list)   :   A list of dictionaries.
                                The dictionary key "tweetTokenized" contains the
                                text to be used.

        Output:
        1. dataset (list)   :   The input dataset with an additional vector representing
                                the frequency of words written with initial caps and all caps
                                in each tweet (normalized by overall number of words), additionally the frequency of
                                capitalized (non-initial) letters in the tweet (normalized by overall number of letters).
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting initial capitalization, words in all cap "
                                                        "and capitalized letters...")
        for tweet in self.dataset:
            firstCaps = 0
            allCaps = 0
            letterCaps = 0
            totalLetters = 0
            totalWords = 0

            for index, word in enumerate(tweet["tweetTokenized"]):
                # if token is a word (we should be careful: assumption that all words only contain letters)
                if word.isalpha():
                    totalLetters += len(word)
                    totalWords += 1
                    if word[0].isupper():
                        firstCaps += 1
                    if word.isupper():
                        if word != "I" and word != "XD":
                            allCaps += 1
                for character in word: #exclude first character, if we do not want that, just word
                    if character.isalpha() and character.isupper():
                        letterCaps += 1
                progressBar.update(1)

            if totalWords != 0:
                tweet["initCapsCount"] = firstCaps / totalWords
                tweet["allCapsCount"] = allCaps / totalWords
            else:
                tweet["initCapsCount"] = 0
                tweet["allCapsCount"] = 0

            if totalLetters != 0:
                tweet["letterCapsCount"] = letterCaps / totalLetters
            else:
                tweet["letterCapsCount"] = 0


            progressBar.close()

    def same_letter_count(self):
        """
        Function to calculate the number of words in each tweet that contain more than 2 consecutively repeating letters.

        Input:
        1. dataset (list)   :   A list of dictionaries.
                                The dictionary key "tweetTokenized" contains the text to be used.

        Output:
        1. dataset (list)   :   The input dataset with the additional an additional key:

                                "repeatingLettersWords":
                                Frequency of words that contain repeating letters (more than 2), normalized by
                                total number of words.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting repeating letters...")
        for tweet in self.dataset:
            repeating_letters_words = 0
            words_count = 0

            for word in tweet["tweetTokenized"]:
                if word.isalpha():
                    words_count += 1
                    if re.search(r'(\w)\1{2,}', word, flags=re.IGNORECASE):
                        repeating_letters_words += 1

            if words_count > 0:
                tweet["repeatingLettersWords"] = repeating_letters_words/words_count
            else:
                tweet["repeatingLettersWords"] = 0
            progressBar.update(1)

        progressBar.close()

    def include_word2vec(self):
        """
            Function to include word2vec word embeddings.


            INFORMATION
            We could try out different models, the higher thhe
            dimension though, the more accurate the results
        """

        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Word Embeddings...")

        # load pre-trained model
        wv = api.load("glove-twitter-100")
        # save the vectors
        # wv.save("./used_models/vectors.kv")
        # wv = "./used_models/vectors.kv"
        for tweet in self.dataset:
            tokenized_sentence = tweet["tweetTokenized"]

            # get dimension of vctor (based on chosen model)
            vector_size = wv.vector_size
            # initialize the vector
            wv_res = np.zeros(vector_size)

            # count the words that are represented as
            # vector --> later for the average of thee sentence
            counter = 0

            for token in tokenized_sentence:
                if token in wv:
                    self.embeddings.append(wv[token])
                    # counter += 1
                    # wv_res += wv[token]
                    # tweet["tweetTokenEmbeddings"].append(wv_res)
                else:
                    self.embeddings.append(None)

            # compute average of the sentence
            # wv_res = wv_res/counter

            progressBar.update(1)

        progressBar.close()

    def question_mark_count(self):
        """
        Function to add a feature representing the number of question marks in each tweet.

        Input:
        dataset (list)  :   A list of dictionaries where each dictionary represents a tweet.

        Output:
        dataset (list)  :   The input dataset with an additional 'questionMarkCount' feature.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting question marks...")
        # by characters:
        for tweet in self.dataset:
            questionCount = tweet["tweetText"].count('?')
            totalChars = len(tweet["tweetText"])
            if totalChars > 0:
                normalized = questionCount / totalChars
            else:
                normalized = 0
            tweet["questionMarkCount"] = normalized
            progressBar.update(1)

        progressBar.close()


    def exclamation_mark_count(self):
        """
        Function to add a feature representing the number of exclamation marks in each tweet.

        Input:
        dataset (list)  :   A list of dictionaries where each dictionary represents a tweet.

        Output:
        dataset (list)  :   The input dataset with an additional 'exclamationMarkCount' feature.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting exclamation marks...")
        # by characters:
        for tweet in self.dataset:
            exclamationCount = tweet["tweetText"].count('!')
            totalChars = len(tweet["tweetText"])
            if totalChars > 0:
                normalized = exclamationCount / totalChars
            else:
                normalized = 0
            tweet["exclamationMarkCount"] = normalized
            progressBar.update(1)

        progressBar.close()

    def period_count(self):
        """
        Function to add a feature representing the number of periods in each tweet.

        Input:
        dataset (list)  :   A list of dictionaries where each dictionary represents a tweet.

        Output:
        dataset (list)  :   The input dataset with an additional 'periodCount' feature.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting periods...")
        # by characters:
        for tweet in self.dataset:
            periodCount = tweet["tweetText"].count('.')
            totalChars = len(tweet["tweetText"])
            if totalChars > 0:
                normalized = periodCount / totalChars
            else:
                normalized = 0
            tweet["periodCount"] = normalized
            progressBar.update(1)

        progressBar.close()

    def find_consecutive_punctuation(self):
        """
        Function to find consecutive punctuation in one tweet.

        Input:
        dataset (list)  :   A list of dictionaries where each dictionary represents a tweet.

        Output:
        dataset (list)  :   The input dataset with an additional 'consecutivePunctuation' feature. For each tweet
                            the value is either 0 (no consecutive punctuation) or 1 (consecutive punctuation present).
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Finding consecutive punctuation...")
        # by characters:
        for tweet in self.dataset:
            consecutive_punctuation = 0
            if re.search(r'[.?!]{2,}', tweet["tweetText"]):
                consecutive_punctuation += 1
            tweet["consecutivePunctuation"] = consecutive_punctuation
            progressBar.update(1)

        progressBar.close()


    def ats_count(self):
        """
        Function to add a feature representing the number of user mentions in each tweet.

        Input:
        dataset (list)  :   A list of dictionaries where each dictionary represents a tweet.

        Output:
        dataset (list)  :   The input dataset with an additional 'taggedCount' feature.
        """
        total_iterations = len(self.dataset)
        progressBar = tqdm(total=total_iterations, desc="Counting @'s...")
        # by characters:
        for tweet in self.dataset:
            tweetText = tweet["tweetText"]
            atsCount = len(re.findall(r'@\w+', tweetText))
            totalChars = len(tweet["tweetText"])
            if totalChars > 0:
                normalized = atsCount / totalChars
            else:
                normalized = 0
            tweet["taggedCount"] = normalized
            progressBar.update(1)

        progressBar.close()

    def create_feature_array(self):
        '''Creates a feature matrix for the dataset object and fills the self.feature_array attribute.'''
        df = pd.DataFrame(self.dataset)
        features = df.drop(columns=[
            "class",
            "tweetText",
            "tweetTokenized",
            "emoticons",
            "emoticons_description",
            "tweetWithoutEmojicon",
            "tweetWithEmojiconDescription",
            "emojis_description",
            "emojis"
        ])
        self.feature_array = features.values

    def create_classification_array(self):
        '''Creates a classification matrix for the dataset object and fills the self.class_array attribute.'''
        df = pd.DataFrame(self.dataset)
        class_pd = df["class"]
        self.class_array = class_pd.values


# Example usage:
preprocessed_tweet = Preprocessing(OG_dataset)
preprocessed_tweet.tokenizer()
preprocessed_tweet.handling_emojis()
preprocessed_tweet.handling_emoticons()
preprocessed_tweet.sentiment()
preprocessed_tweet.create_vocabulary()
preprocessed_tweet.uni_and_bigrams()
#preprocessed_tweet.pos_tagging()
preprocessed_tweet.tweet_length_tokenized()
preprocessed_tweet.caps_counts()
preprocessed_tweet.same_letter_count()
preprocessed_tweet.ats_count()
preprocessed_tweet.exclamation_mark_count()
preprocessed_tweet.question_mark_count()
preprocessed_tweet.period_count()
preprocessed_tweet.find_consecutive_punctuation()
preprocessed_tweet.create_classification_array()
preprocessed_tweet.create_feature_array()

class_array = preprocessed_tweet.class_array
feature_array = preprocessed_tweet.feature_array
