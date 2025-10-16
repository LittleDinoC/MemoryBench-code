# TODO: 在大改代码框架后（domain_and_task）后还没有测试过这个文件，之后将 LoCoMo 合并进来
import os
import json
import importlib
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from src.dataset import BaseDataset
from src.agent import AgentFactory
from src.agent.base_agent import BaseAgent
from src.utils import get_single_dataset


def process_single_data(data, chat_agent: BaseAgent, feedback_agent: BaseAgent, max_rounds: int, dataset: BaseDataset = None):
    """
    针对 data 这组数据，让 feedback_agent(user) 和 chat_agent(assistant) 执行至多 max_rounds 轮对话

    返回值是数据点序号和对话过程
    """

    chat_messages = dataset.get_initial_chat_messages(data["test_idx"])
    implicit_feedback_history = []  # Track implicit feedback for each turn
   
    TRY_TIMES = 3 

    for round_idx in range(max_rounds):
        if round_idx != 0:
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
                agent_response = chat_agent.generate_response(messages=chat_messages)
                break
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
        "dialog": chat_messages,
        "implicit_feedback": implicit_feedback_history,
    }


def save_json_file(save_dir, filename, data):
    with open(os.path.join(save_dir, filename), "w") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def main(args):
    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    assert not args.dataset.startswith("Locomo-") and not args.dataset.startswith("DialSim-"), f"Invalid dataset {args.dataset}"
    dataset = get_single_dataset(args.dataset, args.dataset_config)

    run_config = vars(args)

    # load feedback agent
    with open(args.feedback_agent_config, "r") as fin:
        feedback_agent_config = json.load(fin)
    run_config["feedback_agent_config"] = feedback_agent_config
    feedback_agent = AgentFactory.create(
        method_name="feedback",
        config=feedback_agent_config,
    )

    # load chat agent in different methods
    with open(args.memory_system_config, "r") as fin:
        memory_system_config = json.load(fin)
    run_config["memory_system_config"] = memory_system_config
    memory_system_config["llm_config"]["max_tokens"] = min(
        memory_system_config.get("max_tokens", 2048),
        dataset.max_output_len,
    )
    print("\n", memory_system_config, "\n")
    chat_agent = AgentFactory.create(
        method_name=args.memory_system,
        config=memory_system_config,
    )

    # communication
    sample = args.sample if args.sample != -1 else len(dataset.dataset)
    total_dialogs = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(process_single_data, data, chat_agent, feedback_agent, args.max_rounds, dataset)
            for data in dataset.dataset[:sample]
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            total_dialogs.append(result)
    total_dialogs.sort(key=lambda x: x["test_idx"])

    # save results
    save_dir = os.path.join(
        args.output_dir,
        args.dataset, 
        args.memory_system,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_json_file(save_dir, "run_config.json", run_config)
    save_json_file(save_dir, "dialogs.json", total_dialogs)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/datasets/each.json",
        help="Path to the config file of the dataset.",
    )
    parser.add_argument(
        "--feedback_agent_config", 
        type=str, 
        default="configs/memory_systems/feedback.json",
        help="Path to the config file of the feedback agent.",
    )
    parser.add_argument(
        "--memory_system", # actually the memory agent
        type=str,
        choices=["base"],
        default="base",
        help="baseline name",
    )
    parser.add_argument(
        "--memory_system_config", 
        type=str, 
        default="configs/memory_systems/base.json",
        help="Path to the config file of the chat agent."
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="dialogs/",
        help="Path to save the dialog memory of all sets."
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="max rounds for communication"
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="The number of multithreaded threads."
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=-1,
        help="Use the first sample data points for debugging.",
    )

    args = parser.parse_args()
    assert args.max_rounds >= 1, "max_rounds at least 1"
    print(args)
    main(args)