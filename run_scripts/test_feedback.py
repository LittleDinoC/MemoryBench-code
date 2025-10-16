import json
import subprocess
import os
from termcolor import colored


with open("configs/datasets/each.json", "r") as fin:
    dataset_config = json.load(fin)

def run_script(command):
    command = command.replace("&", "\&")
    print(colored(f"===\n\nRunning command: {command}\n\n===", "blue"))
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(colored(f"Command failed with error: \n\n{e}\n\n\n", "red"))
    print(colored(f"Finished command: {command}\n\n\n", "green"))


for dataset in dataset_config.keys():
    if dataset.startswith("Locomo") or dataset.startswith("DialSim"):
        continue
    command = f"python -m src.test_feedback --dataset {dataset}"
    run_script(command)