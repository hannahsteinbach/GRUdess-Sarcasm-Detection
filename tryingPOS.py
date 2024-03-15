import os
import jpype
from jpype import startJVM, getDefaultJVMPath, JClass, JString, shutdownJVM
from data import load_dataset, cleanURLS, handling_emojis, handling_emoticons

class PosTagger:
    def __init__(self, modelPath):
        startJVM(getDefaultJVMPath())
        self.tagger = JClass('cmu.arktweetnlp.Tagger')(modelPath)
        self.tagger.loadModel(modelPath)

    def tag(self, dataset):
        taggedTweets = []
        for tweet in dataset:
            taggedTweets.append(self.tagger.tokenizeAndTag(JString(tweet)))
        return taggedTweets
    
    def shutdownJVM(self):
        shutdownJVM()
    

def pos_tagging(dataset, modelPath):

    # Load the ARK POS tagger
    posTagger = JClass('cmu.arktweetnlp.Tagger')
    modelPath = './ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'
    tagger = posTagger(JString(modelPath))

    # Tag the tweets
    taggedTweets = []
    for tweet in dataset:
        taggedTweets.append(tagger.tokenizeAndTag(JString(tweet)))

    return taggedTweets

def main():
    dataset = load_dataset('./data/train.En__1_.csv')
    print(jpype.getDefaultJVMPath())
    dataset = cleanURLS(dataset)
    dataset = handling_emojis(dataset)
    model_path = './ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar'
    pos_tagging(dataset, model_path)

if __name__ == '__main__':
    main()