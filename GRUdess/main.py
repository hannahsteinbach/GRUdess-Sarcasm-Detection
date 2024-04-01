"""
    This file is the main file of the project.
    It is designed to lead through all the steps which were involved to get to the
    result of the project.1

    If run, the User can chose the dataset they want to
    see results on and if they want to see it in
        > Training with Cross Validation or
        > Training and Teesting on the test data.

    Afterwards the data will be preprocessed.
    And GRUdess will be executed.

    For more details please refer to the README document.
"""

#####################################
#   modules
#####################################

from extended_data import *
from preprocessing import *
from grudess import *
from focal_loss import *


#####################################
#   functions
#####################################

def run_script():
    """
    First, the data used for the models will be generated.
    Afterwards, a menu pops up, where the user gets guided
    through the implementations and their evaluation - if wanted.
    Input:
        None.
    Output:
        None.
        There will either be printed outputs in the terminal
        (or maybe generated datafiles).
    """

    while True:
        print("""

        Hello, thank you for your interest in GRUdess, 
        a sarcasm identifier.

        There are several options to run the program.

        First, we kindly ask you to choose the dataset you
        want to see the results from.

        Second, you can choose if you want to train and test 
        the data on given train data and test data, or if you
        want to see results of the training with testing
        via Cross Validation.

        To choose options in the following process,
        type in the number in front of the option and hit enter.""")

        # ask user which data should be used
        dataset = int(input(
            "Which data do you want to use? \nPlease choose one of the following: \n0 \tClose program. \n1 \tOG_dataset \n2 \tbalanced_dataset \n3\tunbalanced_dataset\n\n"))

        model = int(input(
            "What do you want to see:\n0\tClose program.\n1\tTraining and Testing on Cross Validation \n2\tTraining and Testing in Test Data\n\n"))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}.")

        num_layers = 12
        hidden_size = 256
        num_classes = 2
        num_epochs = 60
        batch_size = 128
        learning_rate = 0.0001
        k = 3
        max_grad_norm = 5

        # schedular parameters
        scheduler_cooldown = 5
        scheduler_min_lr = 2.000e-7
        scheduler_factor = 0.5
        scheduler_patience = 8

        # early dropout parameters
        patience = 20
        min_delta = 0.005
        dropout_rate = 0.5

        if dataset == 0 or model == 0:
            print("Bye, bye! :)")
            break

        elif dataset not in [1, 2, 3] or model not in [1, 2]:
            print("Your input is invalid. Please try again.")
            continue

        # OG_dataset and CV
        elif dataset == 1 and model == 1:
            # preprocess the dataset
            combined_tensor_train, class_tensor_train = generate_train_tensor(OG_dataset)
            train_dataset = combined_tensor_train
            # run GRUdess
            run_grudess_CV(train_dataset,
                           class_tensor_train,
                           num_layers,
                           hidden_size,
                           num_classes,
                           num_epochs,
                           batch_size,
                           learning_rate,
                           k,
                           max_grad_norm,
                           dropout_rate)

        # OG_dataset and trai/test GRU
        elif dataset == 1 and model == 2:
            # preprocess the dataset
            combined_tensor_train, class_tensor_train, combined_tensor_test, class_tensor_test = generate_train_test_tensor(
                balanced_dataset, testset)
            train_dataset = combined_tensor_train
            test_dataset = combined_tensor_test

            # run GRUdess
            run_grudess_train_test(train_dataset,
                                   class_tensor_train,
                                   test_dataset,
                                   class_tensor_test,
                                   num_layers,
                                   hidden_size,
                                   num_classes,
                                   num_epochs,
                                   batch_size,
                                   learning_rate,
                                   max_grad_norm,
                                   scheduler_cooldown,
                                   scheduler_min_lr,
                                   scheduler_factor,
                                   scheduler_patience,
                                   patience,
                                   min_delta,
                                   dropout_rate)

        # balanced_dataset and CV
        elif dataset == 2 and model == 1:
            # preprocess the dataset
            combined_tensor_train, class_tensor_train = generate_train_tensor(balanced_dataset)
            train_dataset = combined_tensor_train

            # run GRUdess
            run_grudess_CV(train_dataset,
                           class_tensor_train,
                           num_layers,
                           hidden_size,
                           num_classes,
                           num_epochs,
                           batch_size,
                           learning_rate,
                           k,
                           max_grad_norm,
                           dropout_rate)

        # balanced_dataset and train/test GRU
        elif dataset == 2 and model == 2:
            # preprocess the dataset
            combined_tensor_train, class_tensor_train, combined_tensor_test, class_tensor_test = generate_train_test_tensor(
                balanced_dataset, testset)
            train_dataset = combined_tensor_train
            test_dataset = combined_tensor_test

            print("Start Training & Testing")
            # run GRUdess
            run_grudess_train_test(train_dataset,
                                   class_tensor_train,
                                   test_dataset,
                                   class_tensor_test,
                                   num_layers,
                                   hidden_size,
                                   num_classes,
                                   num_epochs,
                                   batch_size,
                                   learning_rate,
                                   max_grad_norm,
                                   scheduler_cooldown,
                                   scheduler_min_lr,
                                   scheduler_factor,
                                   scheduler_patience,
                                   patience,
                                   min_delta,
                                   dropout_rate)

        # unbalanced_dataset and CV
        elif dataset == 3 and model == 1:
            # preprocess the dataset
            combined_tensor_train, class_tensor_train = generate_train_tensor(unbalanced_dataset)
            train_dataset = combined_tensor_train

            # run GRUdess
            run_grudess_CV(train_dataset,
                           class_tensor_train,
                           num_layers,
                           hidden_size,
                           num_classes,
                           num_epochs,
                           batch_size,
                           learning_rate,
                           k,
                           max_grad_norm,
                           dropout_rate)

            # unbalanced_dataset and train/test GRU
        elif dataset == 3 and model == 2:
            # preprocess the dataset
            """
            Disclaimer: We realised too late that before, this function did take the OG dataset
            as well instead of the imbalanced dataset. 
            We changed this now, but our papers are based on these incorrect results (where imbalanced 
            dataset actually is OG dataset). 
            """
            combined_tensor_train, class_tensor_train, combined_tensor_test, class_tensor_test = generate_train_test_tensor(
                unbalanced_dataset, testset)
            train_dataset = combined_tensor_train
            test_dataset = combined_tensor_test

            # run GRUdess
            run_grudess_train_test(train_dataset,
                                   class_tensor_train,
                                   test_dataset,
                                   class_tensor_test,
                                   num_layers,
                                   hidden_size,
                                   num_classes,
                                   num_epochs,
                                   batch_size,
                                   learning_rate,
                                   max_grad_norm,
                                   scheduler_cooldown,
                                   scheduler_min_lr,
                                   scheduler_factor,
                                   scheduler_patience,
                                   patience,
                                   min_delta,
                                   dropout_rate)

        else:
            print("\n??????\n")


if __name__ == "__main__":
    run_script()
