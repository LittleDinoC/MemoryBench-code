import os
import re
import json
import copy
import importlib
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from src.agent import AgentFactory
from src.agent.base_agent import BaseAgent
from src.utils import get_single_dataset, change_dialsim_conversation_to_locomo_form, get_memory_system_config_file

from src.solver import SolverFactory


def process_single_data(data, dataset, solver, feedback_agent: BaseAgent, max_rounds: int):
    chat_messages = dataset.get_initial_chat_messages(data["test_idx"])
    implicit_feedback_history = []  # Track implicit feedback for each turn

    TRY_TIMES = 3

    for round_idx in range(max_rounds):
        if round_idx == 0:
            chat_messages[-1]["origin_content"] = chat_messages[-1]["content"]
        else:
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
                if round_idx == 0:
                    agent_response = solver.agent.generate_response(
                        messages=chat_messages,
                        lang=data["lang"],
                    )
                else:
                    agent_response = solver.agent.llm.generate_response(
                        messages=chat_messages,
                    )
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


def solve_locomo(solver, feedback_agent, dataset, sample, args):
    total_dialogs = []
    solver.memory_locomo_conversation( 
        dataset.conversation,
        session_cnt=dataset.conversation_cnt,
    )

    for i in tqdm(range(sample), desc=f"Chat with {args.memory_system}"):
        data = dataset.dataset[i]
        result = process_single_data(
            data=data,
            dataset=dataset,
            solver=solver,
            feedback_agent=feedback_agent,
            max_rounds=args.max_rounds,
        )
        total_dialogs.append(result)
    return total_dialogs


def solve_dialsim(solver, feedback_agent, dataset, sample, args):
    total_dialogs = []
    conversation, session_cnt = change_dialsim_conversation_to_locomo_form(dataset.corpus)
    solver.memory_dialsim_conversation(conversation, session_cnt)
    for i in tqdm(range(sample), desc=f"Chat with {args.memory_system}"):
        total_dialogs.append(process_single_data(
            data=dataset.dataset[i],
            dataset=dataset,
            solver=solver,
            feedback_agent=feedback_agent,
            max_rounds=args.max_rounds,
        ))
    return total_dialogs


def save_json_file(save_dir, filename, data):
    """
    将数据 data 存储到 save_dir/filename 文件中
    """
    with open(os.path.join(save_dir, filename), "w") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)


def main(args):
    assert args.dataset.startswith("Locomo-") or args.dataset.startswith("DialSim-"), f"Invalid dataset {args.dataset}"

    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
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

    # load memory agent in different methods
    with open(args.memory_system_config, "r") as fin:
        memory_system_config = json.load(fin)
    run_config["memory_system_config"] = memory_system_config
    memory_system_config["llm_config"]["max_tokens"] = min(
        memory_system_config.get("max_tokens", 2048),
        dataset.max_output_len,
    )
    print("\n", memory_system_config, "\n")

    save_dir = os.path.join(
        args.output_dir, 
        args.dataset,
        args.memory_system,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    memory_cache_dir = os.path.join(save_dir, "memory_cache")
    if os.path.exists(memory_cache_dir):
        import shutil
        shutil.rmtree(memory_cache_dir)
    solver = SolverFactory.create(
        method_name=args.memory_system,
        config=memory_system_config,
        memory_cache_dir=memory_cache_dir,
    )

    # communication
    sample = args.sample if args.sample != -1 else len(dataset.dataset)
    
    if args.dataset.startswith("Locomo-"):
        total_dialogs = solve_locomo(solver, feedback_agent, dataset, sample, args)
    else:
        total_dialogs = solve_dialsim(solver, feedback_agent, dataset, sample, args)


    # save results
    save_json_file(save_dir, "run_config.json", run_config)
    save_json_file(save_dir, "dialogs.json", total_dialogs)

    # evaluate first response
    if args.dataset.startswith("Locomo-"):
        predicts = []
        for dialog in total_dialogs:
            resp = ""
            for msg in dialog["dialog"]:
                if msg["role"] == "assistant":
                    resp = msg["content"]
                    break
            predicts.append({
                "test_idx": dialog["test_idx"],
                "response": resp,
            })
        merged_results, detailed_results = dataset.evaluate_and_summary(predicts)
        save_json_file(save_dir, "evaluate_details.json", detailed_results)
        save_json_file(save_dir, "total_results.json", merged_results)



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
        "--memory_system",
        type=str,
        required=True,
        help="baseline name",
    )
    parser.add_argument(
        "--memory_system_config", 
        type=str, 
        # required=True, 
        default=None,
        help="Path to the config file of the chat agent."
    )
    parser.add_argument(
        "--output_dir",
        type=str, 
        default="dialogs/",
        help="Path to save the memory of train sets."
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
        default=-1,
        type=int, 
        help="Use the first sample data points for debugging.",
    )
    args = parser.parse_args()
    assert args.max_rounds >= 1, "max_rounds at least 1"
    assert args.memory_system != "wo_memory", "memory_system should not be wo_memory"
    args.memory_system_config = get_memory_system_config_file(args.memory_system, args.memory_system_config)
    print(args)
    main(args)