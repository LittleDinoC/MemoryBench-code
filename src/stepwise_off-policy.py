import os
import json
import copy
import random
import shutil
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
from argparse import ArgumentParser
from termcolor import colored
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.utils import (
    if_memory_cached, 
    get_single_dataset, 
    get_dataset_series,
    change_dialsim_conversation_to_locomo_form,
    get_memory_system_config_file,
)
from src.solver import SolverFactory

# try:
#     import nltk
#     nltk.data.find('wordnet')
# except LookupError:
#     print("Downloading WordNet data...")
#     nltk.download('wordnet')

def if_rc_dataset(dataset_name):
    if dataset_name.startswith("Locomo-") or dataset_name.startswith("DialSim-"):
        return True
    return False


def build_solver(
    cache_save_dir,
    args,
    copy_from_memory_cache_dir=None,
):
    """
        Build and return a solver instance based on the provided arguments.
        Args:
            cache_save_dir (str): Directory to save the cache.
            args: Parsed command line arguments containing configuration for the solver.
            copy_from_memory_cache_dir (str, optional): Directory to copy the memory cache from.
        Returns:
            solver: An instance of the solver created based on the provided configuration.
            memory_cache_dir (str): Directory where the memory cache is stored.
    """
    memory_cache_dir = os.path.join(
        args.memory_cache_prefix + cache_save_dir, 
        "dataset" if args.dataset else ("domain" if args.domain else "task"),
        args.dataset if args.dataset else (args.domain if args.domain else args.task),
        args.memory_system,
    )
    if copy_from_memory_cache_dir is None:
        if not if_memory_cached(memory_cache_dir) and os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)
    else:
        assert os.path.exists(copy_from_memory_cache_dir), f"Memory cache dir {copy_from_memory_cache_dir} does not exist."
        if os.path.exists(memory_cache_dir):
            shutil.rmtree(memory_cache_dir)
        shutil.copytree(copy_from_memory_cache_dir, memory_cache_dir)
        print(f"Copied memory cache from {copy_from_memory_cache_dir} to {memory_cache_dir}.")
    solver_config = {
        "method_name": args.memory_system,
        "config": args.memory_system_config,
        "memory_cache_dir": memory_cache_dir,
    }
    if args.retrieve_k is not None:
        solver_config["retrieve_k"] = args.retrieve_k
    print("Solver config:", solver_config)
    solver = SolverFactory.create(**solver_config)
    solver.MAX_THREADS = args.threads
    return solver, memory_cache_dir


def load_corpus_to_memory(solver, dataset):
    if dataset.dataset_name.startswith("Locomo-"):
        solver.memory_locomo_conversation( 
            dataset.conversation,
            session_cnt=dataset.conversation_cnt,
        )
    else:
        conversation, session_cnt = change_dialsim_conversation_to_locomo_form(dataset.corpus)
        solver.memory_dialsim_conversation(conversation, session_cnt)


def save_result(data, filename):
    with open(filename, "w") as fout:
        json.dump(data, fout, indent=4, ensure_ascii=False)


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
            "base" if not if_rc_dataset(dataset_name) else args.memory_system,
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
    
    # load configuration
    with open(args.memory_system_config, "r") as fin:
        args.memory_system_config = json.load(fin)
        print(args.memory_system_config)
    
    memory_solver, dialog_memory_cache_dir = build_solver("memory_cache", args, None)
    # memory_solver.create_or_load_memory(total_dialogs, args.dialogs_dir)

    # Load RC dataset corpus first
    for dataset in dataset_lists:
        if dataset.dataset_name.startswith("Locomo") or dataset.dataset_name.startswith("DialSim"):
            print(colored(f"\nLoading corpus of dataset {dataset.dataset_name} to memory...\n", "yellow"))
            load_corpus_to_memory(memory_solver, dataset)
            print(colored(f"\nLoaded corpus of dataset {dataset.dataset_name} to memory.\n", "green"))
    dataset_name_to_class = {dataset.dataset_name: dataset for dataset in dataset_lists}

    # Save path
    name = args.dataset if args.dataset else (args.domain if args.domain else args.task)
    if args.use_delete:
        name += "_with_delete"
    output_dir = os.path.join(
        args.output_dir, 
        "dataset" if args.dataset else ("domain" if args.domain else "task"),
        name,
        args.memory_system, 
        f"start_at_{start_timestamp}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_result(vars(args), os.path.join(output_dir, "run_config.json"))

    # predict test set
    test_predicts = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        print(colored(f"\nPredicting test set of dataset {dataset_name}...", "yellow"))
        predicts = memory_solver.predict_test(dataset, test_ids[dataset_name])
        for pred in predicts:
            pred["dataset"] = dataset_name
            test_predicts.append(pred)
    ccccur_path = os.path.join(output_dir, "begin")
    os.makedirs(ccccur_path, exist_ok=True)
    save_result(test_predicts, os.path.join(ccccur_path, "test_predicts.json"))

    from_idx = 0
    while from_idx < len(total_dialogs):
        to_idx = min(from_idx + args.batch_size, len(total_dialogs))
        cur_dialogs = total_dialogs[from_idx:to_idx]
        print(colored(f"\nProcessing dialogs {from_idx}-{to_idx}... total {len(total_dialogs)}", "yellow"))

        cur_save_dir = os.path.join(
            output_dir, 
            f"[{from_idx},{to_idx})",
        )
        os.makedirs(cur_save_dir, exist_ok=True)
        
        # update memory
        for dia in tqdm(cur_dialogs, desc="Updating memory"):
            memory_solver.agent.add_conversation_to_memory(dia["dialog"], dia["test_idx"])
        save_result(cur_dialogs, os.path.join(cur_save_dir, "training_dialogs.json"))

        # predict test set
        test_predicts = []
        for dataset in dataset_lists:
            dataset_name = dataset.dataset_name
            print(colored(f"\nPredicting test set of dataset {dataset_name}...", "yellow"))
            predicts = memory_solver.predict_test(dataset, test_ids[dataset_name])
            for pred in predicts:
                pred["dataset"] = dataset_name
                test_predicts.append(pred)
        save_result(test_predicts, os.path.join(cur_save_dir, "test_predicts.json"))

        from_idx = to_idx



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
        required=True,
        help="Path to the config file of the dataset.",
    )
    parser.add_argument(
        "--memory_system",
        type=str,
        required=True, 
        help="The memory system to use",
        choices=["wo_memory", "a_mem", "mem0", "memoryos", "bm25_message", "embedder_message", "bm25_dialog", "embedder_dialog"],
    )
    parser.add_argument(
        "--memory_system_config",
        type=str,
        # required=True,
        default=None,
        help="Path to the memory system configuration file",
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
        default="step_off-policy/results/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of threads to use for processing dialogs",
    )
    parser.add_argument(
        "--retrieve_k", 
        type=int,
        default=5,
        help="Number of memories to retrieve for each query",
    ) # 如果 memory_system_config 中有 retrieve_k 这一项，则覆盖

    parser.add_argument(
        "--use_delete", # 仅针对包含 Locomo/DialSim 的 Domain/Task，是否使用 delete api
        action="store_true",
        help="Whether to use delete api when processing Locomo/DialSim datasets",
    )

    parser.add_argument(
        "--memory_cache_prefix",
        type=str,
        default="step_off-policy/",
        help="Prefix path to copy memory cache from",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size of training data for each memory update step",
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
    if args.use_delete:
        if args.domain == "Open-Domain" or args.task == "Long-Short":
            pass
        else:
            assert False, "--use_delete is only applicable for Open-Domain domain or Long-Short task."
        assert args.memory_system not in ["wo_memory", "memoryos"], "--use_delete is not applicable for wo_memory or MemoryOS."
    args.memory_system_config = get_memory_system_config_file(args.memory_system, args.memory_system_config)
    print(args)
    main(args)