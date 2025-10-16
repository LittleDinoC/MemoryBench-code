import json
import subprocess
import os
from termcolor import colored

def run_script(command):
    comand = command.replace("&", "\&")
    print(colored(f"===\n\nRunning command: {command}\n\n===", "blue"))
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(colored(f"Command failed with error: {e}", "red"))
    print(colored(f"Finished command: {command}\n\n===", "green"))



domain = ["Academic\&Knowledge", "Legal", "Open-Domain"]
# task = ["Long-Long", "Short-Short", "Short-Long", "Long-Short"]
for method in ["bm25_message", "bm25_dialog", "embedder_message", "embedder_dialog", "a_mem", "mem0", "memoryos"]:
    for d in domain:
        if d == "Open-Domain" and method == "mem0":
            continue # mem0 does not support Open-Domain
        command = f"python -m src.on-policy --domain {d} --memory_system {method} --dataset_config configs/datasets/domain.json"
        run_script(command)
    # for t in task:
    #     if t == "Long-Short" and method == "mem0":
    #         continue # mem0 does not support Long-Short
    #     command = f"python -m src.stepwise_off-policy --task {t} --memory_system {method}  --dataset_config configs/datasets/task.json"
    #     run_script(command)