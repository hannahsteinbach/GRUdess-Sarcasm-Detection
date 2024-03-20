"""
This file includes the implementation of th "many to one"
GRU model for classification
"""

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#matplotlib inline


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from preprocessing import class_array, feature_array

# feature_tensor = torch.tensor(feature_array, dtype=torch.float32)



# convert tokens into tensors 
# don't need to and can simply use the output of the unigrams 
'''
def pad_inputs(embedding_arrays):
    """
        Function which takes a list of unigrams and pads them
        all to the same length

        Input:
        1. unigrams (list)

        Output:
        1. padded_input (tensor)

        This example works, in case the function does not:
            # test = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([2, 8, 9, 11, 58, 99])]
            # padded_test = pad_sequence(test, batch_first=True)
            # print(padded_test)
    """

    padded_embeddings = list()

    for tweet in embedding_arrays:

        padded_tweet = list()
            
        input_as_tensor = [torch.tensor(element) for element in tweet]

        padded_input = pad_sequence(input_as_tensor, batch_first=True)

        padded_input = pack_padded_sequence(padded_input)

        padded_tweet.append(padded_input)
    
        padded_embeddings.append(padded_tweet)
    
    return padded_embeddings
'''
# have one vector with the embeddings of the tweet-> give it to the 

# 

"""
Tweet : "This is funny :D"

tokens : This, is, funny, :D

embeddings{
	this 	: [0.515456, 0.51514, 0.5441]
	is	: [0.656, 0.54864651, 0.545]
	funny	: [0.789, 0.89562, 0.518]
	:D	: [0.12, 0.156, 0.1458]


features:
	f1: length_of_tweet
	f2: number_of_emoticon
	f3: number_of_initial_capital

feature_vector = [f1=4, f2=1, f3=1]

input_vector = [[embeddings, feature_vector], [same for tweet2]]
--> [
	[0.515456, 0.51514, 0.5441], 
	[0.656, 0.54864651, 0.545], 
	[0.789, 0.89562, 0.518],
	[0.12, 0.156, 0.1458],
	4,
	1,
	f1
]

-->	input_length:   length of the input sequences
--> sequence_length:length of the number of tweets   
"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


# feature_array = pad_inputs(feature_array)

# Network architecture hyperparameters

input_size = len(feature_array[0])   # number of tweets 
seq_length = len(feature_array)
hidden_size = 128
num_layers = 2
num_classes = 2
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Data sets
train_data = None 
test_data = None

# Load data
train_loader = DataLoader(dataset=train_data,
                                           batch_siz=batch_size,
                                           shuffle=False)

test_loader = DataLoader(dataset=test_data,
                                           batch_siz=batch_size,
                                           shuffle=False)

# if the program run too slow:
# change bidirectional into False 

# build RNN together with Patrick Loeber
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        # first arguent of Super is simply the kind
        # of RNN we want; can be interchanged with LSTM, GRU etc.
        super(GRU, self)

        # store number of layers and hidden sizes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # use the rnn class provided by pytorch to create the rnn
        # nn.RNN means that the model is within the nn and is called RNN
        # give it the information it needs
        self.gru =nn.GRU(input_size, hidden_size, num_layers, batch_first=True) 
        # batch_first means that our input (x) has the size of:
        #  batch_size x seq_length x input/feature_size

        # create fully connected (linear) layer
        # use hidden_size as input_size and num_classes as output_size
        # linear is needed to transform results into classes
        # if we want more layers reference Jorge's notebook
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        """
            Based on the documentation, this function needs 2 inputs:
                first one is the input
                second one is the initial hidden state 
        """
    
        # create tensor with only 0's as the initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # number of layers, batch size & hidden_size

        # call out RNN model
        out, _ = self.gru(x, h0) # so input and initial hidden state
        # the output out is of size: batch_siz, seq_length, hidden_size

        # now we want to decode the hidden state only of the last time step; therefore of the form (N,hidden_size)
        out[:, -1, :]

        # need this for the linear layer fc
        out = self.fc(out)

        return out # finito RNN




# initialize the model
model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# criterion & optimizer can be modified
# loss function 
criterion = nn.CrossEntropyLoss()
# optimizer -- Raluca is working on it 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# early stopping - RAAAAAAALUUUUUUUUUUCCAAAAAAAAA


######## train the model ########
n_total_steps = len(train_loader)

"""
def train_model(model, loss, optimizer, num_epochs):

    # save losses over epochs
    train_losses = list()
    val_losses = list()

    # save accuracies over epochs 
    train_accs = list()
    val_accs = list()

    for epoch in range(num_epochs):
        
        # initialize the training
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # loop over train data 
        for tweet, label in zip(feature_array, class_array) : # really look into train_loader
            tweet = tweet.to(device)
            label = label.to(device)


            # forward pass to get prediction of this tweet
            outputs = model(tweet)
            loss = criterion(outputs, label)

            # loss unfction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #loss = # RALUCA LOSS FUNCTION PLEASE WITH (output,labels)

            #    raaaaaaaAAAAAAAAAAAAAAAALUUUUUUUUUCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            # regularization?
            # backward and optimizer #RAAAAAAAAAAAAAAAAAAAAAALUUUUUUUUUUUUUUUUUUUUUCAAAAAAAAAAAAAAAAAAAAAAAAA
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            running_loss += loss.item() * tweet.size(0)

            total_preds += label.size(0)
            correct_preds += (predicted == label).sum().item()

            # include wandb here to track losses etc?
        
        # add the results to the lists from above
        train_losses.append(running_loss / len(feature_array.sampler))
        train_accs.append(100 * correct_preds / total_preds)


        #### validation ####
        # model.eval()
        # val_loss = 0.0
        # correct_preds = 0
        # total_preds = 0

        # # no need for gradient (only for training)

        # # basically the same as for the test func but for validation
        # with torch.no_grad():
        #     # loop over validation data
        #     for tweet, label in val_loader:
        #         tweet = tweet.to(device)
        #         label = label.to(device)

        #         output = model(tweet)

        #         # Rahahahahahahahahahahahahalucaaaaa
        #         #loss = lossfunction

        #         val_loss += loss.item() * label.size()

        #         total_preds += label.size(0)
        #         correct_preds += (output == label).sum().item()

        # val_losses.append(val_loss / len(val_loader.sampler))
        # val_accs.append(100 * correct_preds / total_preds)


        # include the early stopping in case the results get worse 
        # if something:
        #     break

    return train_losses, val_losses, train_accs, val_accs
"""


for epoch in range(num_epochs):

    # loop over train data 
    for index, (tweets, labels) in enumerate(feature_array, class_array) : # really look into train_loader
        tweets = tweets.reshape(-1, seq_length, input_size).to(device)
        labels = labels.to(device)

        # forward pass to get prediction of this tweet
        outputs = model(tweets)
        loss = criterion(outputs, labels)


        # Modification of optimization required :)
        # loss funfction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        # include wandb here to track losses etc?
    
        if (index+1) % 100 == 0:
            print(epoch)

    # add the results to the lists from above
    # train_losses.append(running_loss / len(feature_array.sampler))
    # train_accs.append(100 * correct_preds / total_preds)
        




######## test the model ########
with torch.no_grad(): # look what this is/does
    num_correct = 0
    num_samples = 0

    for tweets, labels in test_loader:
        
        tweets = tweets.reshape(-1, seq_length, input_size).to(device)
        labels = labels.to(device)

        # use trained model on the input tweet
        outputs = model(tweets)

        _, predicted= torch.max(outputs.data, 1) # what does this do?

        num_samples += labels.size(0)
        num_correct += (predicted == labels).sum().item()

    acc = 100.0 * num_correct / num_samples

    print(acc)
