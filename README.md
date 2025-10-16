# MemoryBench

## Introduction

Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. 

Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. 
Experiments show that the effectiveness and efficiency of state-ofthe-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.

This repository is designed to reproducing the results from our paper. It contains our datasets and the code used in all experiments.

## Repository Structure

```plain
baselines/              # Source code of various baselines (modified for stability and vLLM support)
configs/
    datasets/           # Dataset configuration files
    memory_systems/     # Configuration files for different memory system baselines
raw/                    # Raw datasets
run_scripts/            # Scripts for running different experiments
src/                    # Source code for all experiments
    datasets/           # Dataset-related code
    llms/               # Code for LLM deployment
    agent/              # Memory System Agent & User Feedback Simulator
    solver/             # Generation with Memory System Agent
    generate_dialogs/   # Main scripts for dialogue generation
    action_feedback/    # Main scripts using action feedback 
    dataset_statistic.py # Compute dataset statistics
    evaluate.py         # Evaluate results for each dataset
    summary_evaluate_result.py # Summarize and normalize results across datasets
    single_summary.py   # Compute normalized scores based on summary results
    predict.py          # Main script for off-policy experiments
    on-policy.py        # Main script for on-policy experiments
    stepwise_off-policy.py # Main script for stepwise off-policy experiments
    test_feedback.py    # Test feedback-related performance
    train_performance.py# Evaluate training set performance
    utils.py            # Utility functions
```

## Getting Started

### Environment Setup

Use the following commands to set up the conda environment:

```
conda create -n memorybench python=3.10
conda activate memorybench
pip install -r requirements.txt
```

Our experiments use vLLM to deploy LLM services. You can deploy models in a similar way:

```
vllm serve Qwen/Qwen3-32B --port 12345 --chat-template qwen3_nonthinking.jinja     # Qwen3-32B
vllm serve Qwen/Qwen3-8B --port 12366 --chat-template qwen3_nonthinking.jinja      # Qwen3-8B
vllm serve Qwen/Qwen3-Embedding-0.6B --port 12377 --task embed                     # Qwen3-Embedding-0.6B
vllm serve AQuarterMile/WritingBench-Critic-Model-Qwen-7B --port 12388             # LLM as Judge (for WritingBench)
```

If you deploy using these commands, you don't need to modify configuration files in `configs/memory_systems`. Otherwise, adjust the configuration files based on your own setup. Model configuration details can be found in `src/llms/`.

We use Qwen3’s non-thinking mode via the official vLLM configuration. You can find more details in the [documents](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes).

Please set up the `.env` file to specify evaluation models and optional OpenAI API configurations.
These evaluation models are used for all LLM-as-judge evaluations and integrated scoring across multiple metrics.

### Dataset and Memory System Configuration

Dataset configurations are located in `configs/datasets/`:

+ each.json — metadata for each dataset

+ domain.json — datasets grouped by domain

+ task.json — datasets grouped by task

Since the LoCoMo dataset includes 10 conversations, we split them into 10 separate datasets (locomo-0 to locomo-9) for convenience.

For each memory system, the correspondence between paper names, code names, and configuration files (in the `configs/memory_systems`) is shown below:

| Paper Name |	Code Name	| Config File |
|--------------|----------------|----------------|
| Vanilla  | wo_memory           | base.json |
| BM25-M | bm25_message       | bm25.json |
| BM25-S | bm25_dialog | bm25.json |
| Emb-M | embedder_message  | embedder.json |
| Emb-S | embedder_dialog   | embedder.json |
| A-Mem | a_mem | a_mem.json |
| Mem0 | mem0 | mem0.json |
| MemoryOS | memoryos | memoryos.json |

You can modify these configuration files to adjust parameters of each memory system.
Their implementations can be found under `src/agent/`.


### Dialogues Generation

You can directly run `run_scripts/create_dialogs.py` to generate dialogues for each dataset.  
Alternatively, you can use our released dialog files — just place them in the `dialogs/` directory.

For datasets without a corpus, the dialogues are between a simple LLM Agent and the
user feedback simulator.  
The main script is `src/generate_dialogs/basic.py`, which by default uses configurations `base.json` and `feedback.json` in the `configs/memory_systems/` directory for the simple LLM Agent and the Feedback Agent, respectively. You can specify the dataset name with `--dataset`.

Example command:

```bash
python -m src.generate_dialogs.basic --dataset JRE-L
```

For datasets with a corpus (e.g., LoCoMo and DialSim), each memory system runs independently.
LLMsys will first memorize the corpus and then have a dialogue with the user feedback simulator.
The main script is `src/generate_dialogs/reading.py`.
You need to specify the memory system using `--memory_system` (matching a config file in `configs/memory_systems/`) and the dataset name using `--dataset`.

Example command:

```bash
python -m src.generate_dialogs.reading --memory_system bm25_message --dataset locomo-0
```

### Generation

We provide various experiment scripts in the `run_scripts/` directory.
Running them will produce generation results for each data point.
Evaluation instructions are described in the Evaluation section below.

Below is a brief introduction to each experiment.

#### Off-policy Experiments

Run all off-policy experiments directly using:

```bash
bash run_scripts/off_policy.sh
```

Since the Mem0 method takes too long to run on Open-Domain and LiSo tasks,
it is skipped by default — you can enable it manually if needed.

The main script is `src/predict.py`.
Specify the memory system (`--memory_system`),
and the domain/task using `--domain` or `--task`.
The domain/task configuration file (configs/datasets/) should be specified via `--dataset_config`.

Example command:

```bash
python -m src.predict --memory_system bm25_message --domain Open-Domain --dataset_config configs/datasets/domain.json
```

You can find more detailed parameter configuration in the code, including the number of single retrievals (`--retrieve_k`, default is 5), etc. The results are stored in `off-policy/results` by default.

#### Stepwise Off-policy Experiments

Run all stepwise off-policy experiments with:

```bash
bash run_scripts/stepwise_off_policy.sh
```

The main script is `src/stepwise_off_policy.py`, sharing the same configuration options as the off-policy setup mostly.
Specifically, you can specify the `--batch_size` of dialogues to be memorized in a single step, which defaults to 100.
The results are stored in `step_off-policy/results` by default.

Example command:

```bash
python -m src.stepwise_off-policy --memory_system bm25_message --domain Open-Domain --dataset_config configs/datasets/domain.json
```

#### On-policy Experiments

Run all on-policy experiments with:

```bash
bash run_scripts/on_policy.sh
```

The main script is `src/on_policy.py`, sharing the same configuration options as the off-policy setup mostly.
You can also set `--max_rounds` to specify the number of conversation rounds (default is 3), `--batch_size` to specify the number of conversations to remember at a time (default is 100), and `--step` to specify the number of runs (default is 10)
The results are stored in `on-policy/results` by default.

Example command:

```bash
python -m src.stepwise_off-policy --memory_system bm25_message --domain Open-Domain --dataset_config configs/datasets/domain.json
```

#### Train Performance

You can measure training set performance via:

```bash
python run_scripts/train_performance.py
```

This experiment shares all configurations with off-policy experiments,
using the same dialogues — the only difference is that answers are generated for the questions in the training set.

The main script is `src/train_performance.py` and the results are stored in `train_performance/results` by default.

#### Testing the Effectiveness of Feedback

This experiment compares the performance with and without feedback for each data point.

For each data point and its dialogue:

+ The assistant’s first reply is treated as the no-feedback response.
+ The dialogue is then re-run with the feedback included, producing the with-response result.

The comparison between the two is used to show how understanding feedback can improve performance.

You can run this experiment by `python run_scripts/test_feedback.py`.

By default, the dialogues stored in `dialogs/` are used, and `configs/memory_systems/base.json` is used as the agent configuration file. You can specify the dataset name using `--dataset`, and the results are stored in `test_feedback/results`.

Notably, this experiment directly evaluates the inference results at the end of the run, and you can see the comparison results in `compare.json` in the experiment directory.

Example command:

```bash
python -m src.test_feedback --dataset JRE-L
```

#### Use Feedback with Action

We also evaluated the effect of using Action signals from Feedback as the criterion for memory selection.  
The corresponding scripts can be found in the `action_feedback/` directory.

Specifically, for all memory systems, we used dialogs (excluding those in the test set) that contained a specific action within the feedback as the memory source.  
We then tested the system’s performance when it had memorized these dialogs.  
The inference code for memory systems in this setup is located in `predict_with_implicit_feedback.py`.

In particular, we also experimented with SFT-based memory learning.  
In this case, we only considered the first turn of each dialog and selected those containing a specific action for training.  
The training script is `train_sft_lora.py`, and the inference script is `predict_sft.py`.

You can specify the action type using the `--action_feedback` argument,  
which supports two options: `like` or `copy`.

### Evaluation

The evaluation process consists of two stages.  
First, each dataset is evaluated individually using its own metrics to obtain detailed scores for every data point.  
Then, the results across datasets are integrated to produce overall scores for each domain or task.

For the first stage, you can use `src/evaluate.py` to evaluate all runs within a specific `results/` folder.  
This will generate detailed evaluation results for each data point, stored in subdirectories as `evaluate_details.json`.

Example command:

```bash
python -m src.evaluate --result_path off-policy/results/
```

For the second stage, we use min-max normalization to standardize scores across datasets.
The min and max values were determined from the off-policy experiments (note that they might not be the absolute min and max under all settings).
These serve as the normalization reference for computing the scores of other experiments.

You can either directly use our normalization configuration
(found in `configs/final_evaluate_summary_wo_details.json`),
or recompute the min and max based on your own experimental results.

To recompute the min and max values for a given results directory:

```bash
python -m src.summary_evaluate_result --result_path xxx/results
```

To directly aggregate the evaluation details into final normalized scores:

```bash
python -m src.single_summary --result_path xxx/results
```

This command will produce the final scores for all experiments in the directory, stored as `summary.json` in each subfolder.
By default, it uses our provided normalization reference.
You can specify your own using `--default_min_max_path <your_min_max_config.json>`.
