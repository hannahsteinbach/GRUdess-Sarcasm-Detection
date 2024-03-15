"""
    Within this file we write the script for the evaluation.
    Plan is to be able to run multiple models after
    each other and let wandb generate a good overview of the
    results for us.

"""


# pip install wandb

import wandb

# open file with wandb api
with open("./evaluation/wandbAPI.txt", "r") as api_file:
    api_key = api_file.read().strip()

# initialize wanb with api // log into wandb
wandb.login(key="API_KEY")
