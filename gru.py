import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#matplotlib inline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


"""
Let's talk about the input


"""


def pad_inputs(preprocessed_input):
    """
        Function which padds our input data to the same lenght 

        Input:
        1. preprocessed_input (list)

        Output:
        1. padded_input (tensor)

        This example works, in case the function does not:
            # test = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([2, 8, 9, 11, 58, 99])]
            # padded_test = pad_sequence(test, batch_first=True)
            # print(padded_test)
    """
    input_as_tensor = [torch.tensor(element) for element in preprocessed_input]

    padded_input = pad_sequence(input_as_tensor, batch_first=True)

    padded_input = pack_padded_sequence(padded_input)

    return padded_input


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)   # GRU
        self.fc = nn.Linear(hidden_size, num_classes)  
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Set initial hidden and cell states - GRU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        out = self.softmax(out) 
        return out   


