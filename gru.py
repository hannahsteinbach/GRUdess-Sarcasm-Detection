import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from preprocessing import combined_tensor_train, class_tensor_train
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

all_f1_scores = []


class GRUDESS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUDESS, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # x -> batch_size, sequence_len, input_size

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # x.size(0) = batch_size

        # call rnn model
        out, _ = self.gru(x, h0)  # (N, 1, 128)
        # shape of output: batch_size, seq_length, hidden_size
        # wan:out (N, 128)
        out = out[:, -1, :]  # (N, 128)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
input_size = combined_tensor_train.size(1)
sequence_len = 1
num_layers = 2

hidden_size = 128
num_classes = 2
num_epochs = 25
batch_size = 50
learning_rate = 0.005

k = 3

# Data sets
train_dataset = TensorDataset(combined_tensor_train,
                              class_tensor_train)
test_data = None

# include k-fold Cross Validation
skf = StratifiedKFold(n_splits=k,
           shuffle=True)

"""
TO DO:
Switch from kFold to StrasifiedkFold, is better for inbalanced dataset
https://www.youtube.com/watch?v=-8s9KuNo5SA&t=416s
min 19:00
"""

# Train the model
# n_total_steps = len(train_loader)

# Track accuracy for each fold
folds_values = dict()

for fold, (train_index, val_index) in enumerate(skf.split(combined_tensor_train, class_tensor_train)):
    print(f"Fold {fold + 1}/{k}")

    # save results of losses etc. for every fold/epoch etc.
    folds_values[fold] = {
        "losses": list(),
        "train_sarcasm": 0,
        "val_sarcasm": 0,
    }
    folds_values[fold]["f1_scores"] = []

    # subset dataset based on folds
    train_subset = Subset(train_dataset,
                          train_index)
    val_subset = Subset(train_dataset,
                        val_index)

    # create dataloader
    train_loader = DataLoader(train_subset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_subset,
                            batch_size=batch_size,
                            shuffle=True)

    model = GRUDESS(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for i, (tweets, labels) in enumerate(train_loader):
            # want size [batch, seq, input]
            # -> 25, 1, 13
            tweets = tweets.reshape(-1, sequence_len, input_size).to(device)
            labels = labels.to(device)

            for val in labels:
                if val == 1:
                    folds_values[fold]["train_sarcasm"] += 1

            # forward pass
            outputs = model(tweets)
            loss = criterion(outputs, labels)
            folds_values[fold]["losses"].append(loss)

            # backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")

    model.eval()
    n_correct = 0
    n_samples = 0
    # validation test the model
    with torch.no_grad():

        for tweets, labels in val_loader:
            tweets = tweets.reshape(-1, sequence_len, input_size).to(device)
            labels = labels.to(device)
            outputs = model(tweets)

            for val in labels:
                if val == 1:
                    folds_values[fold]["val_sarcasm"] += 1

            _, predicted = torch.max(outputs.data, 1)
            print("predicted:", predicted)
            print("labels:", labels)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            # Compute F1 score
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())
            folds_values[fold]["f1_scores"].append(f1)
            all_f1_scores.append(f1)

        acc = 100.0 * n_correct / n_samples
        """TO DO: TAKE VALIDATION AVERAGE 
        SAV RESULTS"""

overall_f1_score = np.mean(all_f1_scores)
print("Overall F1 score:", overall_f1_score)
print(f"Accuracy of {acc}%")

# test the model
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0

#     for tweets, labels in val_loader:
#         tweets = tweets.reshape(-1, sequence_len, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(tweets)

#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f"Accuracy of {acc}%")
