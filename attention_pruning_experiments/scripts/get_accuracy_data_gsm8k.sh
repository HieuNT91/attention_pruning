#!/bin/bash

# Define the list of model repositories
model_repos=(
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
)

num_heads_in_group_values=(1)
# pruned_layer_values=(20 25 15 0)
pruned_layer_values=(25 20 15 10 5 0)
script_path="shapley_prune/get_accuracy_data_gsm8k.py"

for model_repo in "${model_repos[@]}"; do
    for num_heads_in_group in "${num_heads_in_group_values[@]}"; do
        for pruned_layer in "${pruned_layer_values[@]}"; do
            echo "Running with model_repo: $model_repo, num_heads_in_group: $num_heads_in_group, pruned_layer: $pruned_layer"
            accelerate launch $script_path \
                --model_repo "$model_repo" \
                --num_heads_in_group "$num_heads_in_group" \
                --pruned_layer "$pruned_layer" \
                --batch_size 64 \
                --save_every_n_gens 3 \
                --dataset test \
                --project_id gsm8k_test_1000 \
                --num_samples -1
        done
    done
done

echo "All runs completed!"