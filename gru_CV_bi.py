import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from focalloss import class_weights, FocalLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from preprocessing import combined_tensor_train, class_tensor_train
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score


class GRUDESS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUDESS, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # x -> batch_size, sequence_len, input_size

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device)  # x.size(0) = batch_size; *2 for bidirectional

        # call rnn model
        out, _ = self.gru(x, h0)  # (N, 1, 128)
        # shape of output: batch_size, seq_length, hidden_size
        # wan:out (N, 128)
        # out = out[:, -1, :]  # (N, 128)
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_grudess_CV(train_dataset,
                   class_tensor_train,
                   num_layers,
                   hidden_size,
                   num_classes,
                   num_epochs,
                   batch_size,
                   learning_rate,
                   k):
    all_precision_scores = list()
    all_recall_scores = list()
    all_f1_scores = []

    # Define hyperparameters
    input_size = combined_tensor_train.size(1)
    sequence_len = 1
    num_classes = 2

    # Data sets
    train_dataset = TensorDataset(combined_tensor_train,
                                  class_tensor_train)

    # include k-fold Cross Validation
    skf = StratifiedKFold(n_splits=k,
                          shuffle=True)

    # Train the model

    # Track accuracy for each fold
    folds_values = dict()

    for fold, (train_index, val_index) in enumerate(skf.split(combined_tensor_train, class_tensor_train)):
        print(f"Fold {fold + 1}/{k}")

        # save results of losses etc. for every fold/epoch etc.
        folds_values[fold] = {
            "losses": list(),
            "train_sarcasm": 0,
            "val_sarcasm": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0,
            "recall": 0,
            "f1_scores": list()
        }

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

        weights = torch.FloatTensor(class_weights(class_tensor_train))

        model = GRUDESS(input_size, hidden_size, num_layers, num_classes).to(device)

        criterion = FocalLoss(alpha=weights, gamma=2)
        # criterion = nn.CrossEntropyLoss()  -> choose with loss function to use
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            loss_e = 0
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
                loss_e += loss.item()
                folds_values[fold]["losses"].append(loss)

                # backward and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Loss: {loss.item()}")

                    # Print the average loss for the epoch
            average_epoch_loss = loss_e / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_epoch_loss}")

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
                # print("predicted:", predicted)
                # print("labels:", labels)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for y_gold, y_pred in zip(labels, predicted):
                    if y_pred == 1 and y_pred == y_gold:
                        folds_values[fold]["tp"] += 1

                    elif y_pred == 1 and y_pred != y_gold:
                        folds_values[fold]["fp"] += 1

                    elif y_pred == 0 and y_pred != y_gold:
                        folds_values[fold]["fn"] += 1

                    elif y_pred == 0 and y_pred == y_gold:
                        folds_values[fold]["tn"] += 1

                # compute precision
                try:
                    prec = (folds_values[fold]["tp"]) / (folds_values[fold]["tp"] + folds_values[fold]["fp"])
                except:
                    prec = 0
                folds_values[fold]["precision"] = prec
                all_precision_scores.append(prec)

                # compute recall
                try:
                    rec = (folds_values[fold]["tp"]) / (folds_values[fold]["tp"] + folds_values[fold]["fn"])
                except:
                    rec = 0
                folds_values[fold]["recall"] = rec
                all_recall_scores.append(rec)

                # Compute F1 score
                f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())
                folds_values[fold]["f1_scores"].append(f1)
                all_f1_scores.append(f1)

            acc = 100.0 * n_correct / n_samples

    overall_prec_score = np.mean(all_precision_scores)
    overall_rec_score = np.mean(all_recall_scores)
    overall_f1_score = np.mean(all_f1_scores)

    print("Overall Precision score:", overall_prec_score)
    print("Overall Recall score:", overall_rec_score)
    print("Overall F1 score:", overall_f1_score)
    print(f"Accuracy of {acc}%")


train_dataset = combined_tensor_train
class_tensor_train = class_tensor_train
num_layers = 2

hidden_size = 128
num_classes = 2
num_epochs = 25
batch_size = 50
learning_rate = 0.0001

k = 3

run_grudess_CV(train_dataset,
               class_tensor_train,
               num_layers,
               hidden_size,
               num_classes,
               num_epochs,
               batch_size,
               learning_rate,
               k)
