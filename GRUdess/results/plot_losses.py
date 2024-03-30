"""
    This file plots the average losses during the training
    of GRUdess based on the three datasets OG, BD and ID.
"""

import json
import matplotlib.pyplot as plt

def plot_losses(jsonfile1, jsonfile2, jsonfile3):
    """
    Function which plots the average losses of
    the given input files: OG BD and ID.

    Input:
    1. json_file1 (str)  : Path to the json file.
    2. json_file2 (str)  : Path to the json file.
    3. json_file3 (str)  : Path to the json file.

    Output:
    1. Graphics.
    """

    with open(jsonfile1, "r", encoding="UTF-8") as file:
        results_OG = json.load(file)

        # averages losses per epoch
        averages_losses_OG = results_OG["average_losses"]

    with open(jsonfile2, "r", encoding="UTF-8") as file:
        results_BD = json.load(file)

        # averages losses per epoch
        averages_losses_BD = results_BD["average_losses"]

    with open(jsonfile3, "r", encoding="UTF-8") as file:
        results_ID = json.load(file)

        # averages losses per epoch
        averages_losses_ID = results_ID["average_losses"]

    # plot losses
    plt.figure(figsize=(10, 6))

    plt.plot(range(len(averages_losses_OG)), averages_losses_OG, marker='o', color='violet', label='Average Losses Trained on OG')

    plt.plot(range(len(averages_losses_BD)), averages_losses_BD, marker='o', color='lightcoral', label='Average Losses Trained on BD')

    plt.plot(range(len(averages_losses_ID)), averages_losses_ID, marker='o', color='lightgreen', label='Average Losses Trained on ID')

    # set x-axis and labels
    plt.xticks(range(max(len(averages_losses_OG), len(averages_losses_BD), len(averages_losses_ID))))

    # set y-axis and labels
    plt.yticks([i * 0.1 for i in range(int(max(max(averages_losses_OG), max(averages_losses_BD), max(averages_losses_ID)) * 10) + 1)])

    # labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Average Losses Over Epochs')
    plt.legend()

    plt.grid(True)
    plt.show()

    
        
# plot losses:
jsonfile1 = "./results/OG_results/OG_results_of_GRUdess_traintest.json"
jsonfile2 = "./results/balanced_dataset/BD_results_of_GRUdess_traintest.json"
jsonfile3 = "./results/imbalanced_results/ID_results_of_GRUdess_traintest.json"

plot_losses(jsonfile1, jsonfile2, jsonfile3)