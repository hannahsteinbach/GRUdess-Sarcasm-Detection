# The WitCraft Wortjongleurinnen Sarcasm Detector for Tweets

Welcome! This project aims to detect sarcasm in text data gathered from Twitter by employing a bidirectional Gated Recurrent Unit (GRU) recurrent neural network. Our approach is unique in the sense that it combines both pre-trained word and emoji embeddings as well as carefully selected features that are meant to represent various linguistically-grounded components of sarcastic statements. 💗🧙🏼‍♀️

## Our features

| Classification                        | Feature                                                     | Corresponding Functions in preprocessing.py       |
|---------------------------------------|-------------------------------------------------------------|---------------------------------|
| **I. Token-specific features**        |                                                             |                                 |
| Lexical features                      | Unigrams                                                    | `create_vocabulary` + `create_uni_bigram_tensor `              |
|                                       | Bigrams                                                     | `create_bigrams()` + `create_uni_bigram_tensor `                 |
| Syntactic features                    | POS-tags                                                    | `pos_tagging()`                |
| Embeddings                            | Word and Emoji Embeddings                                   | `include_2vec()` + `create_embedding_tensor`  |
| Sentiment/emotion features            | Token sentiments                                            | `create_valence_tensor()`        |
| **II. Tweet-specific features**       |                                                             |                                 |
| Sentiment/emotion features            | Polarity of a tweet                                         | `sentiment()`          |
|                                       | Frequency of words containing repeating letter sequences    | `same_letter_count()`   |
|                                       | Sentiment before consecutive punctuation                     | `valence_before_repeating_punctuation()` |
|                                       | Hyperboles                                                  | `detect_hyperbole()`              |
| Stylistic/Sentiment/emotion features | Frequency of words written in all-caps                      | `caps_counts()`           |
|                                       | Frequency of capitalized letters                            | `caps_counts()` |
| Pragmatic features                    | Tweet length                                                | `tweet_length_tokenized()`            |
|                                       | Frequency of user mentions                                  | `ats_count()`      |
|                                       | Frequency of emojis                                         | `handling_emojis()`             |
|                                       | Frequency of emoticons                                      | `handling_emoticons()`          |
| Punctuation features                  | Frequency of question marks                                 | `question_mark_count()`     |
|                                       | Frequency of exclamation marks                              | `exclamation_mark_count()`  |
|                                       | Frequency of periods                                        | `period_count()`            |
|                                       | Consecutive punctuation                       | `find_consecutive_punctuation()` |
| Stylistic features                    | Words with first letter capitalized                        | `get_first_letter_capitalized_words()` |

All tweet specific features are stored in a feature tensor by calling the function `create_feature_tensor()`. Additionally, there are distinct tensors for each token-specific tensor (for unigrams, bigrams, POS-tags, embeddiings and token sentiments) as listed in the table. Note that the order of the execution of the functions is important since some features depend on others. It is recommended to keep the order given in our code files for any dataset "dataset":

- `data.tokenizer()`
- `dataset.tweet_length_tokenized()`
- `dataset.include_2vec()`
- `dataset.create_embedding_tensor()`
- `dataset.handling_emojis()`
- `dataset.handling_emoticons()`
- `dataset.pos_tagging()`
- `dataset.sentiment()`
- `dataset.caps_counts()`
- `dataset.same_letter_count()`
- `dataset.ats_count()`
- `dataset.exclamation_mark_count()`
- `dataset.question_mark_count()`
- `dataset.period_count()`
- `dataset.find_consecutive_punctuation()`
- `dataset.create_valence_tensor()`
- `dataset.valence_before_repeating_punctuation()`
- `dataset.detect_hyperbole()`
- `dataset.create_vocabulary()`
- `dataset.create_bigrams()`
- `dataset.create_uni_bigram_tensor()`
- `dataset.create_feature_tensor()`
- `dataset.create_classification_tensor()`




## Getting Started

# 1. Install dependencies

Make sure to first meet all of the requirements for installed dependencies first. Here is a list of all of the packages that you must have installed on your device in order for the code to run successfully:
- numpy
- torch
- sklearn
- wandb
- tqdm 
- somajo 
- someweta
- nltk.sentiment.vader 
- emoji
- gensim
- pandas

______

## More specific:
click==8.1.7
colorama==0.4.6
contourpy==1.2.0
cycler==0.12.1
emoji==2.11.0
filelock==3.13.3
fonttools==4.50.0
fsspec==2024.3.1
gensim==4.3.2
Jinja2==3.1.3
joblib==1.3.2
kiwisolver==1.4.5
MarkupSafe==2.1.5
matplotlib==3.8.3
mpmath==1.3.0
networkx==3.2.1
nltk==3.8.1
numpy==1.26.4
packaging==24.0
pandas==2.2.1
pillow==10.2.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
regex==2023.12.25
scikit-learn==1.4.1.post1
scipy==1.12.0
six==1.16.0
sklearn==0.0
smart-open==7.0.4
SoMaJo==2.4.2
SoMeWeTa==1.8.1
sympy==1.12
threadpoolctl==3.4.0
torch==2.2.2
tqdm==4.66.2
typing_extensions==4.10.0
tzdata==2024.1
wrapt==1.16.0

## Or install all libraries with the provided requireents.txt file within GRUdess.


## Usage (Train and Test GRUdess)
To train the model and see results in general please follow the following steps:

1. Download the folder GRUdess and save it somewhere on your device.
2. Install all necessary libraries as they are listed in "1. Install dependencies".
3. Run main.py (in your terminal). It starts the program and will lead you through the following possible options.
- After running main.py you will be asked to chose one of the following datasets **OG_dataset**, **Balanced_dataset** or **Inbalanced_dataset**. Select the one you want by inserting the number in front of it and hit "enter".
- Afterwards you will be asked if you (1) want to train GRUdess with Cross Validation or if you (2) want to train GRUdess on the whole chosen train data and test it on the test data.
- After selecting your preferences, GRUdess will start and you can watch the show :)

## Authors \& Support
Contact any of the three Wortjongleurinnen at {rilla ; steinbach4 ; yuezuencueoglu}@uni-potsdam.de


## License
This project is not licensed.

## Datasets statistic

The following datasets are used for this classification task: 

- OG_dataset (as provided by Farha et al. (2022))
 
Overall: 4335 tweets → 867 sarcastic, 3468 non-sarcastic (including rephrases)

- Balanced, extended dataset (OG_dataset, extended by data from Ghosh and Veale (2016))
	 
Overall: 6936 tweets → 3468 sarcastic, 3468 non-sarcastic (including rephrases)

- Imbalanced, extended dataset (OG_dataset, extended by data from Ghosh and Veale (2016))
	
Overall: 8670 tweets → 1734 sarcastic, 6936 non-sarcastic (including rephrases)

- Test set (as provided by Farha et al. (2022))

Overall: 1400 tweets → 200 sarcastic, 1200 non-sarcastic
