# pip install wandb

import wandb

# open file with wandb api
with open("./evaluation/wandbAPI.txt", "r") as api_file:
    api_key = api_file.read().strip()

wandb.login(key="API_KEY")