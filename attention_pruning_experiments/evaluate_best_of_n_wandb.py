import wandb 
import os 
import json 
import numpy as np

def get_file_from_wandb(file_name, wandb_run_id, root_dir="tmp"):
    api = wandb.Api()
    run = api.run(wandb_run_id) # wandb_run_id have format f"{entity}/{project}/{run_id}"
    run.file(file_name).download(replace=True, root=root_dir)
    with open(os.path.join(root_dir, file_name), "r") as f:
        data = json.load(f)
    return data

def get_accuracy_matrix(head_accuracies):
    ordered_heads =  [f"no_prune_acc_vec_{i}" for i in range(0, len(head_accuracies.keys()))]
    accuracy_matrix = []
    for head in ordered_heads:
        accuracy_matrix.append(head_accuracies[head])
    return np.array(accuracy_matrix).T

def evaluate_best_of_n(acc_mat, n=16):
    for i in range(1, n+1):
        top_k_accuracies = acc_mat[:, :i]
        correct_predictions = np.any(top_k_accuracies == 1, axis=1)
        accuracy = correct_predictions.mean() * 100
        print(f"Top {i} | Accuracy = {accuracy:.2f}% +-{correct_predictions.std():.2f}%")
    
    
wandb_run_id = "hieunt91/best_of_n_debug_test/ngzle8wb"
data = get_file_from_wandb("wandb-summary.json", wandb_run_id=wandb_run_id)
acc_mat = get_accuracy_matrix(data['head_accuracies'])
evaluate_best_of_n(acc_mat, n=16)