# The WitCraft Wortjongleurinnen Sarcasm Detector for Tweets

Welcome! This project aims to detect sarcasm in text data gathered from Twitter by employing a bidirectional Gated Recurrent Unit (GRU) recurrent neural network. Our approach is unique in the sense that it combines both pre-trained word and emoji embeddings as well as carefully selected features that are meant to represent various linguistically-grounded components of sarcastic statements. 💗🧙🏼‍♀️

## Our hand-crafted features classified

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

# 2. Train and evaluate model

To train the model and see results in general please follow the following steps:
1. Download the folder GRUdess and save it somewhere on your device.
2. Install all necessary libraries as they are listed in "1. Install dependencies".
3. Run main.py (in your terminal). It starts the program and will lead you through the following possible options.
- After running main.py you will be asked to chose one of the following datasets **OG_dataset**, **Balanced_dataset** or **Inbalanced_dataset**. Select the one you want by inserting the number in front of it and hit "enter".
- Afterwards you will be asked if you (1) want to train GRUdess with Cross Validation or if you (2) want to train GRUdess on the whole chosen train data and test it on the test data.
- After selecting your preferences, GRUdess will start and you can watch the show :)


## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Authors \& Support
This project is creaContact any of the three Wortjongleurinnen at {rilla ; steinbach4 ; yuezuencueoglu}@uni-potsdam.de

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## License
This project is not licensed.

## Datasets statistic

The following datasets are used for this classification task: 

- OG_dataset (as provided by Abu Farha et al. (2022))
 
Overall: 4335 tweets → 867 sarcastic, 3468 non-sarcastic (including rephrases)

- Balanced, extended dataset (OG_dataset, extended by data from Ghosh and Veale (2016))
	 
Overall: 6936 tweets → 3468 sarcastic, 3468 non-sarcastic (including rephrases)

- Imbalanced, extended dataset (OG_dataset, extended by data from Ghosh and Veale (2016))
	
Overall: 8670 tweets → 1734 sarcastic, 6936 non-sarcastic (including rephrases)

- Test set (as provided by Abu Farha et al. (2022))

Overall: 1400 tweets → 200 sarcastic, 1200 non-sarcastic
