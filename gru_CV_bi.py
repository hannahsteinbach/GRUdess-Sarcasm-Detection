import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from focalloss import class_weights, FocalLoss
from torch.utils.data import  DataLoader, TensorDataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from preprocessing import combined_tensor_train, class_tensor_train
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import wandb
random.seed(9)

wandb.init(project="gru_test")
config = wandb.config
config.scheduler_factor = 0.1
config.scheduler_patience = 3
config.scheduler_cooldown = 2
config.scheduler_min_lr = 1e-6
config.earlystop_patience = 5
config.earlystop_min_delta = 0.001
config.num_layers = 2
config.hidden_size = 128
config.dropout_rate = 0.2
config.num_epochs = 25
config.batch_size = 50
config.learning_rate = 0.001
config.max_grad_norm = 5


scheduler_factor = config.scheduler_factor
scheduler_patience = config.scheduler_patience
scheduler_cooldown = config.scheduler_cooldown
scheduler_min_lr = config.scheduler_min_lr
earlystop_patience = config.earlystop_patience
earlystop_min_delta = config.earlystop_min_delta
num_layers = config.num_layers
hidden_size = config.hidden_size
dropout_rate = config.dropout_rate
num_epochs = config.num_epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
max_grad_norm = config.max_grad_norm
num_classes = 2



class GRUdess(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout_rate):  # typical dropout values around 0.2-0.5; experimentation needed
        super(GRUdess, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # x -> batch_size, sequence_len, input_size

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirectional
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            device)  # x.size(0) = batch_size; *2 for bidirectional

        # call rnn model
        out, _ = self.gru(x, h0)  # (N, 1, 128)
        # shape of output: batch_size, seq_length, hidden_size
        # wan:out (N, 128)
        # out = out[:, -1, :]  # (N, 128)
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class EarlyStopping:

    def __init__(self, patience=earlystop_patience,
                 min_delta=earlystop_min_delta):  # for now, we can set these as defaults, but we will probably have to adjust it after we notice how our model trains

        self.patience = patience
        self.minDelta = min_delta
        self.counter = 0
        self.bestLoss = None
        self.stopped = False

    def __call__(self, lossValue):
        if self.bestLoss == None:
            self.bestLoss = lossValue
        elif self.bestLoss - lossValue > self.minDelta:
            self.bestLoss = lossValue
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early stopping!')
                self.stopped = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_grudess_CV(train_dataset,
                   class_tensor_train,
                   num_layers,
                   hidden_size,
                   num_classes,
                   num_epochs,
                   batch_size,
                   learning_rate,
                   k, max_grad_norm):

    all_precision_scores = list()
    all_recall_scores = list()
    all_f1_scores = []
    all_accuracy_scores = []

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

    weights = torch.FloatTensor(class_weights(class_tensor_train))

    model = GRUdess(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = FocalLoss(alpha=weights, gamma=2)
    # criterion = nn.CrossEntropyLoss()  -> choose with loss function to use
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                  patience=scheduler_patience, cooldown=scheduler_cooldown,
                                  min_lr=scheduler_min_lr)

    earlyStop = EarlyStopping(patience=earlystop_patience, min_delta=earlystop_min_delta)

    for fold, (train_index, val_index) in enumerate(skf.split(combined_tensor_train, class_tensor_train)):
        wandb.log({"Fold": fold + 1})
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

                # gradient clipping:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # backward and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Loss: {loss.item()}")
            average_epoch_loss = loss_e / len(train_loader)
            scheduler.step(average_epoch_loss) # i (hannah) filled this, is that correct?
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_epoch_loss}")

            wandb.log({
                "Average epoch loss": average_epoch_loss,
            })

            # if the model performance stops improving before iterating through all epochs,
            # stop training early
            earlyStop(folds_values[fold]["losses"][-1])  # check if the last loss value is the best, need to access
                                                         #  correct entry fri
            if earlyStop.stopped:
                print("Early stopping!")
                break #indented so it only stops when it should


        model.eval()
        n_correct = 0
        n_samples = 0
        # validation test the model
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

                # Log evaluation metrics
                acc = 100.0 * n_correct / n_samples     #where does this go??
                all_accuracy_scores.append(acc)

                wandb.log({
                    "loss": loss.item(),
                    "recall": rec,
                    "precision": prec,
                    "f1_score": f1,
                    "accuracy": acc
                })

    overall_prec_score = np.mean(all_precision_scores)
    overall_rec_score = np.mean(all_recall_scores)
    overall_f1_score = np.mean(all_f1_scores)
    overall_accuracy_score = np.mean(all_accuracy_scores)


    print("Overall Precision score:", overall_prec_score)
    print("Overall Recall score:", overall_rec_score)
    print("Overall F1 score:", overall_f1_score)
    print(f"Overall accuracy of {overall_accuracy_score}%")

    wandb.log({
        "Overall Precision": overall_prec_score,
        "Overall Recall": overall_rec_score,
        "Overall F1 Score": overall_f1_score,
        "Overall accuracy": acc
    })

    wandb.finish()


train_dataset = combined_tensor_train
class_tensor_train = class_tensor_train

k = 3

run_grudess_CV(train_dataset,
               class_tensor_train,
               num_layers,
               hidden_size,
               num_classes,
               num_epochs,
               batch_size,
               learning_rate,
               k,
               max_grad_norm)
