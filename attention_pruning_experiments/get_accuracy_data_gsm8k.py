import wandb
import numpy as np
from src.transformers_utils import IntervenableTransformers
from src.gsm8k_utils import *
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
parser.add_argument("--n_clusters", type=int, default=5)
parser.add_argument("--subsample_size", type=int, default=100)
parser.add_argument("--save_every_n_gens", type=int, default=5)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--dataset", type=str, default='test') # choice of 'train', 'test', 'test_updated'
parser.add_argument("--project_id", type=str, default='test') 
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
    "num_return_sequences": 1,
    "use_cache": True,
    "do_sample": False,
    "batch_size_per_device": args.batch_size,
}



if model.distributed_state.is_main_process:
    wandb.login()
    wandb.init(
        project=f"our_method_{args.project_id}_{args.dataset}",
        name=f"gsm8k_l_{args.pruned_layer}_nheads_{args.num_heads_in_group}_nsam_{args.num_samples}_model_{args.model_repo.split('/')[-1]}",
        config=vars(args),
    )
    wandb.config.update(config_args)

# Load dataset
# data = load_subsampled_data(subsampled_file_path)
if args.dataset == 'train':
    questions, answers = get_gsm8k_dataset(split='train')
    model.distributed_state.print(f"Loaded subsampled data from with {len(questions)} samples")
else:
    questions, answers  = get_gsm8k_dataset(split='test')
    model.distributed_state.print(f"Loaded test_updated data with {len(questions)} samples")

if args.num_samples > 0:
    questions = questions[:args.num_samples]
    answers = answers[:args.num_samples]

evaluator = GSM8kEvaluator()
head_groups_to_prune = [list(range(group_start_idx, group_start_idx + args.num_heads_in_group)) for group_start_idx in range(0, model.model.config.num_attention_heads, args.num_heads_in_group)]
no_prune_accs = []
head_accuracies = {}

all_prompts = [apply_prompt(sample, args.model_repo) for sample in questions]
golden_answers = [sample for sample in answers]

if model.distributed_state.is_main_process:
    wandb.log({'dataset': {
        "prompts": all_prompts,
        "golden_answers": golden_answers
    }})
    print(f"Total {len(all_prompts)} prompts!")

if args.dataset == 'train':
    prompt_cache_path = f"shapley_prune/tmp/cache/gsm8k/{args.model_repo.split('/')[-1]}_ns{args.num_samples}_sene{args.save_every_n_gens}_bz{args.batch_size}_subsample{args.subsample_size}_cluster{args.n_clusters}/no_prune_{args.dataset}"
else:    
    prompt_cache_path = f"shapley_prune/tmp/cache/gsm8k/{args.model_repo.split('/')[-1]}_ns{args.num_samples}_sene{args.save_every_n_gens}_bz{args.batch_size}/no_prune_{args.dataset}"
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
    no_prune_accs, no_prune_mean, no_prune_std = evaluator.get_success_rate(
                model_outputs, golden_answers, config_args["num_return_sequences"], stat_track=False
            )
    head_accuracies["no_prune_acc_vec"] = no_prune_accs.copy()
    wandb.log({"no_prune_all_acc_mean": no_prune_mean, "no_prune_all_acc_std": no_prune_std})
    model.distributed_state.print({"no_prune_all_acc_mean": no_prune_mean, "no_prune_all_acc_std": no_prune_std})

for group_idx, head_group in enumerate(head_groups_to_prune):
    head_group_str = "".join(map(str, head_group))
    if args.dataset == 'train':
        prompt_cache_path = f"shapley_prune/tmp/cache/{args.model_repo.split('/')[-1]}_ns{args.num_samples}_pl{args.pruned_layer}_sene{args.save_every_n_gens}_bz{args.batch_size}_subsample{args.subsample_size}_cluster{args.n_clusters}/{head_group_str}_{args.dataset}"
    else:
        prompt_cache_path = f"shapley_prune/tmp/cache/{args.model_repo.split('/')[-1]}_ns{args.num_samples}_pl{args.pruned_layer}_sene{args.save_every_n_gens}_bz{args.batch_size}/{head_group_str}_{args.dataset}"
    model_outputs = model.generate(
        inputs=all_prompts,
        config=config_args,
        return_raw_output=False,
        heads_to_prune=head_group,
        layers_to_prune=[args.pruned_layer],
        stat_track=False,
        prompt_cache_path=prompt_cache_path,
        save_every_n_gens=args.save_every_n_gens,
        use_prompt_cache=True
    )
    
    if model.distributed_state.is_main_process:
        head_group_accs_, mean_acc, std_acc = evaluator.get_success_rate(
                model_outputs, golden_answers, config_args["num_return_sequences"], stat_track=False
            )
        head_accuracies[f"{head_group_str}_acc_vec"] = head_group_accs_.copy()
        wandb.log({
            f"heads_{head_group}_all_acc_mean": mean_acc,
            f"heads_{head_group}_all_acc_std": std_acc
        })
        model.distributed_state.print({
            f"heads_{head_group}_all_acc_mean": mean_acc,
            f"heads_{head_group}_all_acc_std": std_acc
        })
        wandb.log({"head_accuracies": head_accuracies})
    
if model.distributed_state.is_main_process:
    wandb.finish()