# The WitCraft Wortjongleurinnen Sarcasm Detector for Tweets

Welcome! This project aims to detect sarcasm in text data gathered from Twitter by employing a bidirectional Gated Recurrent Unit (GRU) recurrent neural network. Our approach is unique in the sense that it combines both pre-trained word and emoji embeddings as well as carefully selected features that are meant to represent various linguistically-grounded components of sarcastic statements. 💗🧙🏼‍♀️

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

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
