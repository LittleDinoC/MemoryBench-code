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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.agent import AgentFactory
from src.agent.base_agent import BaseAgent
from src.solver import SolverFactory
from src.solver.base import BaseSolver
from src.predict import load_corpus_to_memory


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


def predict_single_data(
    solver: BaseSolver,
    feedback_agent: BaseAgent,
    dataset: BaseDataset,
    data,
    max_rounds: int,
    training_set_idx: int,
) -> Dict[str, Dict]:
    """
        Predict a single data point using the provided solver and dataset.
        Args:
            solver (BaseSolver): The solver instance to use for prediction.
            dataset (BaseDataset): The dataset instance containing the data.
            data (dict): The data point to predict.

        Returns:
            dict: A dictionary containing the prediction results and dialogs.
    """
    TRY_TIMES = 3

    chat_messages = dataset.get_initial_chat_messages(data["test_idx"])
    implicit_feedback_history = []
    predict_ret = solver.predict_single_data(
        dataset=dataset,
        data=data
    )
    chat_messages.append({
        "role": "assistant",
        "content": predict_ret["response"],
    })
    
    for round_idx in range(max_rounds-1):
        for cnt in range(TRY_TIMES):
            try:
                if_stop, user_feedback, implicit_action = feedback_agent.get_feedback(
                    messages=chat_messages, 
                    data=data,
                    dataset_instance=dataset,
                )
                break
            except Exception as e:
                print(e)
                continue
        else:
            break
        # Store implicit feedback for this turn
        implicit_feedback_history.append({
            "round": round_idx,
            "implicit_action": implicit_action.value,
            "terminated": if_stop
        })
        if if_stop:
            break
        chat_messages.append({
            "role": "user",
            "content": user_feedback,
        })
        for cnt in range(TRY_TIMES):
            try:
                agent_response = solver.agent.llm.generate_response(
                    messages=chat_messages,
                )
            except Exception as e:
                print(e)
                continue
        else:
            break
        chat_messages.append({
            "role": "assistant",
            "content": agent_response,
        })
    return {
        "test_idx": data["test_idx"],
        "dataset": dataset.dataset_name,
        "training_set_idx": training_set_idx,
        "dialog": chat_messages,
        "implicit_feedback": implicit_feedback_history,
    }


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
    train_data_details = []
    for dataset in dataset_lists:
        dataset_name = dataset.dataset_name
        name_to_ids = dataset.get_test_ids(
            truncate_size=dataset.sample_count,
            test_ratio=0.2,
        )
        test_ids[dataset_name] = name_to_ids["test"]
        for train_ids in name_to_ids["train"]:
            train_data_details.append({
                "dataset": dataset_name,
                "test_idx": train_ids,
            })
        output_str = "Loaded {} dialogs from dataset {} and use {} data for testing".format(len(name_to_ids["train"]), dataset_name, len(name_to_ids["test"]))
        print(colored(output_str, "yellow"))
    print(colored("Loaded total {} train {} test".format(len(train_data_details), sum([len(v) for v in test_ids.values()])), "yellow"))
    random.seed(42)
    
    # load feedback agent
    with open(args.feedback_agent_config, "r") as fin:
        feedback_agent_config = json.load(fin)
    args.feedback_agent_config = feedback_agent_config
    feedback_agent = AgentFactory.create(
        method_name="feedback",
        config=feedback_agent_config,
    )

    # load memory system
    with open(args.memory_system_config, "r") as fin:
        args.memory_system_config = json.load(fin)
        print(args.memory_system_config)
    memory_solver, dialog_memory_cache_dir = build_solver(f"memory_cache/{start_timestamp}", args, None)

    # Load RC dataset corpus first
    for dataset in dataset_lists:
        if dataset.dataset_name.startswith("Locomo") or dataset.dataset_name.startswith("DialSim"):
            print(colored(f"\nLoading corpus of dataset {dataset.dataset_name} to memory...\n", "yellow"))
            load_corpus_to_memory(memory_solver, dataset)
            print(colored(f"\nLoaded corpus of dataset {dataset.dataset_name} to memory.\n", "green"))
    dataset_name_to_class = {dataset.dataset_name: dataset for dataset in dataset_lists}


    # Save path
    name = args.dataset if args.dataset else (args.domain if args.domain else args.task)
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

    for cur_step in range(args.step):
        print(colored("\n\nStep {}/{}:".format(cur_step+1, args.step), "blue", attrs=["bold", "underline"]))
        cur_save_dir = os.path.join(
            output_dir, 
            f"step_{cur_step}",
        )
        os.makedirs(cur_save_dir, exist_ok=True)

        # 从 train_data_details 中选 batch_size 个数据进行 memory update
        train_ids = [i for i in range(len(train_data_details))]
        sample_train_ids = random.sample(train_ids, min(args.batch_size, len(train_ids)))

        # generating training dialogues
        def solve_train(sample_idx):
            data = train_data_details[sample_idx]
            dataset = dataset_name_to_class[data["dataset"]]
            return predict_single_data(
                solver=memory_solver,
                feedback_agent=feedback_agent,
                dataset=dataset,
                data=dataset.dataset[data["test_idx"]],
                max_rounds=args.max_rounds,
                training_set_idx=sample_idx,
            )
        training_dialogs = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(solve_train, sample_idx) for sample_idx in sample_train_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating training dialogues"):
                ret = future.result()
                training_dialogs.append(ret)
        save_result(training_dialogs, os.path.join(cur_save_dir, "train_dialogs.json"))
        
        # update memory
        for dia in tqdm(training_dialogs, desc="Updating memory"):
            memory_solver.agent.add_conversation_to_memory(dia["dialog"], dia["test_idx"])

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
        "--feedback_agent_config", 
        type=str, 
        default="configs/memory_systems/feedback.json",
        help="Path to the config file of the feedback agent.",
    )
    parser.add_argument(
        "--memory_system",
        type=str,
        required=True, 
        help="The memory system to use",
        choices=["a_mem", "mem0", "memoryos", "bm25_message", "embedder_message", "bm25_dialog", "embedder_dialog"],
    )
    parser.add_argument(
        "--memory_system_config",
        type=str,
        # required=True,
        default=None,
        help="Path to the memory system configuration file",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="on-policy/results/",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="max rounds for communication"
    )
    parser.add_argument(
        "--retrieve_k", 
        type=int,
        default=5,
        help="Number of memories to retrieve for each query",
    ) # 如果 memory_system_config 中有 retrieve_k 这一项，则覆盖

    parser.add_argument(
        "--memory_cache_prefix",
        type=str,
        default="on-policy/",
        help="Prefix path to copy memory cache from",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size of training data for each memory update step",
    )
    parser.add_argument(
        "--step", 
        type=int, 
        default=10,
        help="Number of update steps",
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
    args.memory_system_config = get_memory_system_config_file(args.memory_system, args.memory_system_config)
    print(args)
    main(args)