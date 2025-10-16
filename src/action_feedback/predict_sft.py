import os
import json
import copy
import random
import shutil
from tqdm import tqdm
import requests
from datetime import datetime
from typing import List, Dict
from argparse import ArgumentParser
from termcolor import colored
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.utils import (
    if_memory_cached, 
    get_single_dataset, 
    get_dataset_series,
    change_dialsim_conversation_to_locomo_form
)
from src.solver import SolverFactory


def main(args):
    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")

    if args.dataset:
        dataset_lists = [get_single_dataset(args.dataset, args.dataset_config)]
        dataset_lists[0].sample_count = None
    else:
        dataset_lists = get_dataset_series(
            domain_or_task_name=args.domain if args.domain else args.task,
            config_path=args.dataset_config,
        )

    # split train, test
    test_ids = {}
    total_dialogs = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        name_to_ids = dataset.get_test_ids(
            truncate_size=dataset.sample_count,
            test_ratio=0.2,
        )
        test_ids[dataset_name] = name_to_ids["test"]
        dialog_file = os.path.join(
            args.dialogs_dir, 
            dataset_name, 
            "base",
            "dialogs.json"
        )
        print(colored(f"Loading dialogs from {dialog_file}", "yellow"))
        assert os.path.exists(dialog_file), f"Dialog file {dialog_file} not found."
        with open(dialog_file, "r") as fin:
            _dialogs = json.load(fin)
            for dia in _dialogs:
                if dia["test_idx"] in name_to_ids["train"]:
                    total_dialogs.append(dia)
        print("Loaded {} dialogs from dataset {} and use {} data for testing".format(len(name_to_ids["train"]), dataset_name, len(name_to_ids["test"])))
    print(f"Loaded {len(total_dialogs)} dialogs for memory creation.")
    random.seed(42)
    random.shuffle(total_dialogs)

    name = args.dataset if args.dataset else (args.domain if args.domain else args.task)
    output_dir = os.path.join(
        args.output_dir, 
        "domain",
        name,
        "sft",
        # f"start_at_{start_timestamp}"
        str(args.learning_rate),
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "run_config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    
    vllm_config = json.load(open("configs/memory_systems/base.json"))
    solver = SolverFactory.create(
        method_name="wo_memory",
        config=vllm_config,
    )
    solver.MAX_THREADS = args.threads
    
    ok_cnt = 0
    for action in ["like", "copy"]:
        for epoch in range(1, 6):
            lora_path = os.path.join(
                f"./action_feedback/sft_models/lr={args.learning_rate}/", 
                args.domain, 
                f"lora_r8_alpha32_{action}_only",
                f"epoch={epoch}_ckpt"
            )
            if not os.path.exists(lora_path):
                print(colored(f"LoRA path {lora_path} not found, skipping.", "yellow"))
                continue
            
            save_path = os.path.join(output_dir, action, f"epoch={epoch}")
            if os.path.exists(os.path.join(save_path, "predict.json")):
                print(colored(f"Prediction file {os.path.join(save_path, 'predict.json')} already exists, skipping.", "blue"))
                ok_cnt += 1
                continue
            
            # add lora adapter to vllm
            url = vllm_config["llm_config"]["vllm_base_url"] + "/load_lora_adapter"
            headers = {"Content-Type": "application/json"}
            data = {
                "lora_name": "sft_adapter",
                "lora_path": lora_path
            }
            
            response = requests.post(url, headers=headers, json=data)
            assert response.status_code == 200, f"Failed to load LoRA adapter: {response.text}"
            print(colored(f"Loaded LoRA adapter from {lora_path}", "green"))
            total_predicts = []
            for dataset in dataset_lists:
                dataset_name = dataset.dataset_name
                print(f"Evaluating dataset {dataset_name} with {len(test_ids[dataset_name])} test data.")
                predicts = solver.predict_test(dataset, test_ids[dataset_name])
                for pred in predicts:
                    pred["dataset"] = dataset_name
                    total_predicts.append(pred)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "predict.json"), "w") as fout:
                json.dump(total_predicts, fout, indent=4, ensure_ascii=False)
            print(colored(f"Saved predictions to {save_path}", "green"))

            # unload lora adapter
            url = vllm_config["llm_config"]["vllm_base_url"] + "/unload_lora_adapter"
            headers = {"Content-Type": "application/json"}
            data = {
                "lora_name": "sft_adapter",
            }
            response = requests.post(url, headers=headers, json=data)
            assert response.status_code == 200, f"Failed to unload LoRA adapter: {response.text}"
            print(colored(f"Unloaded LoRA adapter sft_adapter", "green"))

    return ok_cnt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        help="dataset name",
    ) 
    parser.add_argument(
        "--domain",
        type=str,
        # required=True,
        help="domain name",
    )
    parser.add_argument(
        "--task",
        type=str,
        # required=True,
        help="task name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/datasets/domain.json",
        help="Path to the config file of the dataset.",
    )
    parser.add_argument(
        "--dialogs_dir",
        type=str,
        default="dialogs/",
        help="Directory containing dialog files",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="action_feedback/results/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of threads to use for processing dialogs",
    )
    # parser.add_argument(
    #     "--memory_cache_prefix",
    #     type=str,
    #     default="",
    #     help="Prefix path to copy memory cache from",
    # )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
    )

    args = parser.parse_args()
    # dataset, domain, task 三个只能有一个
    cnt = 0
    if args.dataset is not None:
        cnt += 1
    if args.domain is not None:
        cnt += 1
    if args.task is not None:
        cnt += 1
    assert cnt == 1, "Only one of --dataset, --domain, --task can be specified." 
    print(args)
    main(args)