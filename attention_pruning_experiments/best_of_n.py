import wandb
import numpy as np
from src.transformers_utils import IntervenableTransformers
from src.math500_utils import *
from src.utils import set_seed
from src.grading import grader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_repo", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
parser.add_argument("--pruned_layer", type=int, default=27)
parser.add_argument("--num_heads_in_group", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--temperature", type=float, default=0.9)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--n_clusters", type=int, default=5)
parser.add_argument("--num_return_sequences", type=int, default=16)
parser.add_argument("--subsample_size", type=int, default=100)
parser.add_argument("--save_every_n_gens", type=int, default=5)
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--dataset", type=str, default='test') # choice of 'train', 'test', 'test_updated'
parser.add_argument("--project_id", type=str, default='debug') 
args = parser.parse_args()

set_seed(args.seed)
# Initialize model
model = IntervenableTransformers(
    model_repo=args.model_repo,
    task_type='causal_lm',
    use_auto_model=False,
    logger=None,
    use_accelerate=True,
)

config_args = {
    "attention_implementation": "flash_attention_2",
    "max_new_tokens": args.max_new_tokens,
    "repetition_penalty": 1.0,
    "num_return_sequences": args.num_return_sequences,
    "temperature": args.temperature,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "use_cache": True,
    "do_sample": True,
    "batch_size_per_device": args.batch_size,
}


if model.distributed_state.is_main_process:
    wandb.login()
    wandb.init(
        project=f"best_of_n_{args.project_id}_{args.dataset}",
        name=f"nsam_{args.num_samples}_model_{args.model_repo.split('/')[-1]}",
        config=vars(args),
    )
    wandb.config.update(config_args)

# Load dataset
# data = load_subsampled_data(subsampled_file_path)
if args.dataset == 'train':
    subsampled_file_path = f"shapley_prune/tmp/subsample{args.subsample_size}_cluster{args.n_clusters}.jsonl"
    data = load_dataset(subsampled_file_path)
    model.distributed_state.print(f"Loaded subsampled data from {subsampled_file_path} with {len(data)} samples")
else:
    data = load_dataset('datasets/math500/test.jsonl')
    model.distributed_state.print(f"Loaded test_updated data with {len(data)} samples")

if args.num_samples > 0:
    data = data[:args.num_samples]

evaluator = Math500Evaluator()
head_groups_to_prune = [list(range(group_start_idx, group_start_idx + args.num_heads_in_group)) for group_start_idx in range(0, model.model.config.num_attention_heads, args.num_heads_in_group)]
no_prune_accs = []
head_accuracies = {}

all_prompts = [apply_prompt(sample, args.model_repo) for sample in data]
all_subjects = [sample['subject'] for sample in data]
golden_answers = [sample['answer'] for sample in data]

if model.distributed_state.is_main_process:
    wandb.log({'dataset': {
        "prompts": all_prompts,
        "subjects": all_subjects,
        "golden_answers": golden_answers
    }})
    print(f"Total {len(all_prompts)} prompts!")
    
prompt_cache_path = f"shapley_prune/tmp/cache/best_of_n_{args.model_repo.split('/')[-1]}_ns{args.num_samples}_sene{args.save_every_n_gens}_bz{args.batch_size}/no_prune_{args.dataset}"
model_outputs = model.generate(
    inputs=all_prompts,
    config=config_args,
    return_raw_output=False,
    heads_to_prune=[],
    layers_to_prune=[],
    stat_track=False,
    prompt_cache_path=prompt_cache_path,
    save_every_n_gens=args.save_every_n_gens,
    use_prompt_cache=True
)

if model.distributed_state.is_main_process:
    for i in range(config_args["num_return_sequences"]):
        no_prune_accs, no_prune_mean, no_prune_std = evaluator.get_success_rate(
                    model_outputs[i::config_args["num_return_sequences"]], 
                    golden_answers, 
                    1, stat_track=False
                )
        head_accuracies[f"no_prune_acc_vec_{i}"] = no_prune_accs.copy()
        wandb.log({f"no_prune_all_acc_mean_{i}": no_prune_mean, f"no_prune_all_acc_std_{i}": no_prune_std})
        model.distributed_state.print({f"no_prune_all_acc_mean_{i}": no_prune_mean, f"no_prune_all_acc_std_{i}": no_prune_std})
        
    wandb.log({"head_accuracies": head_accuracies})
    wandb.log({"model_outputs": model_outputs})
    wandb.finish()
