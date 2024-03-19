import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#matplotlib inline


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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

input_vector = [embeddings, feature_vector]
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


# convert tokens into tensors 
# don't need to and can simply use the output of the unigrams 

def pad_inputs(unigrams):
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
    input_as_tensor = [torch.tensor(element) for element in unigrams]

    padded_input = pad_sequence(input_as_tensor, batch_first=True)

    padded_input = pack_padded_sequence(padded_input)

    return padded_input

device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
print("Device:", device)



# Network architecture hyperparameters

input_size = len(unigrams)   # size of vocabulary
hidden_size = 128
num_layers = 2
num_classes = 2

# if the program run too slow:
# change bidirectional into False 

# build RNN together with Patrick Loeber
class RNN(nn.Module):
    # first arguent of Super is simply the kind
    # of RNN we want; can be interchanged with LSTM, GRU etc.
    super(RNN, self)

# initialize the model
model = RNN()

class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)   # GRU
        self.fc = nn.Linear(hidden_size, num_classes)  
        self.softmax = nn.LogSoftmax(dim=1)
        # make it bidirectional
        self.bidirectional = bidirectional
        

    def forward(self, x):
        # Set initial hidden and cell states - GRU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        out = self.softmax(out) 
        return out


#################### TRAINING THE NETWORK ####################
    
# optimization method
# Learn on plateau
# early stopping - as soon as the model stops improving/gets better it stops
    # Jorge has a function for this he'll provide Raluca 
# track losses over all epochs and note them down (with wandb?)
    
# initiate the model
model = RNN_GRU(input_size, hidden_size, num_layers, num_classes) 
