import wandb 
import os 
import json 
import numpy as np
import matplotlib.pyplot as plt
from src.math500_utils import load_dataset
import seaborn as sns

import pandas as pd 

REDOWNLOAD = False 

LLAMA_8B_WANDB_RUNS = [
    "revise_test_500_test/runs/0817w719", # 31
    "revise_test_500_test/runs/x9bsv50u", # 5
    "revise_test_500_test/runs/zci6pd00", # 10
    "revise_test_500_test/runs/lefvps6r", # 15
]

QWEN_MATH_1d5B_WANDB_RUNS = [
    "our_method_test_500_test/runs/5ymzrz1y", # 0
    "our_method_test_500_test/runs/uq77aobw", # 5
    "our_method_test_500_test/runs/k86jhjz6", # 10
    "our_method_test_500_test/runs/s03ilwjt", # 15
    "our_method_test_500_test/runs/798q235q", # 20
    "revise_test_500_test/runs/0pzyzihk", # 27
]

QWEN_MATH_7B_WANDB_RUNS = [
    "revise_test_500_test/runs/0sd0n22i", # 0
    "revise_test_500_test/runs/q2poyi11", # 5
    "revise_test_500_test/runs/kwy7kexv", # 15
    "revise_test_500_test/runs/fxxjdlze", # 27
]

QWEN_BASE_7B_WANDB_RUNS = [
    "revise_test_500_test/runs/06xzeepk", # 0
    "revise_test_500_test/runs/m9c0dh88", # 5
    "revise_test_500_test/runs/esdbghl8", # 15
    "revise_test_500_test/runs/l393skcq", # 27
]


def get_file_from_wandb(wandb_run_id, root_dir="tmp/results"):
    os.makedirs(root_dir, exist_ok=True)
    file_names = ["wandb-summary.json", "wandb-metadata.json"]
    api = wandb.Api()
    run = api.run(wandb_run_id) # wandb_run_id have format f"{entity}/{project}/{run_id}"

    for file_name in file_names:
        run.file(file_name).download(replace=True, root=root_dir)

        with open(os.path.join(root_dir, file_name), "r") as f:
            if file_name == "wandb-summary.json":
                summary = json.load(f)
            else:
                metadata = json.load(f)
        
    model_name =metadata['args'][metadata['args'].index('--model_repo')+1]
    model_name = model_name.split("/")[1]
    pruned_layer =metadata['args'][metadata['args'].index('--pruned_layer')+1]
    rename_to = f"{model_name}_{pruned_layer}.json"
    
    os.rename(os.path.join(root_dir,file_names[0]), os.path.join(root_dir,rename_to))

def get_file_from_cache(model_repo, root_dir="tmp/results"):
    summaries = []
    prune_layers = []
    
    for filename in os.listdir(root_dir):
        if filename.startswith(model_repo) and filename.endswith('.json'):
            full_path = os.path.join(root_dir, filename)
            try:
                pruned_layer = filename[len(model_repo)+1:-5]  # +1 for underscore, -5 for .json
                with open(full_path, 'r') as f:
                    summary = json.load(f)
                    summaries.append(summary)
                    prune_layers.append(pruned_layer)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")
    
    return summaries, prune_layers

def get_accuracy_matrix(head_accuracies):
    ordered_heads =  [f"{i}_acc_vec" for i in range(0, len(head_accuracies.keys())-1)] + ["no_prune_acc_vec"]
    accuracy_matrix = []
    for head in ordered_heads:
        accuracy_matrix.append(head_accuracies[head])
    return np.array(accuracy_matrix).T

    

dataset = load_dataset('../datasets/math500/test.jsonl')
# wandb_run_id = "our_method_test_500_test/798q235q"
# wandb_run_id = "revise_test_500_test/runs/wrj028fh"

if REDOWNLOAD:
    for wandb_run_id in QWEN_MATH_1d5B_WANDB_RUNS:
        data = get_file_from_wandb(wandb_run_id=wandb_run_id)


model_repo = "Qwen2.5-Math-1.5B-Instruct"
summaries, prune_layers = get_file_from_cache(model_repo, root_dir="tmp/results")


for summary, pl in zip(summaries, prune_layers):
    acc_mat = get_accuracy_matrix(summary['head_accuracies'])
    subjects = [sample['subject'] for sample in dataset]
    methods =  [f"Prune Head {i}" for i in range(0, len(summary['head_accuracies'].keys())-1)] + ["No Prune"]
    unique_subjects = sorted(set(subjects))
    num_subjects = len(unique_subjects)
    # Replace long subject names with short forms

    # Step 2: Sort unique_subjects by the last row of acc_by_subject
    # Initialize an accuracy matrix for methods x unique_subjects
    num_subjects = len(unique_subjects)
    acc_by_subject = np.zeros((len(methods), num_subjects))  # Example placeholder

    # Calculate accuracy as before
    for subject_idx, sub in enumerate(unique_subjects):
        subject_mask = np.array(subjects) == sub
        subject_acc = acc_mat[subject_mask, :]
        if subject_acc.shape[0] > 0:
            acc_by_subject[:, subject_idx] = subject_acc.mean(axis=0)
        else:
            acc_by_subject[:, subject_idx] = np.nan

    # Sort the unique_subjects based on the last row of acc_by_subject
    # Get the indices of the sorting order based on the last row (descending order)
    sorting_indices = np.argsort(acc_by_subject[-1])
    unique_subjects = [unique_subjects[i] for i in sorting_indices]
    acc_by_subject = acc_by_subject[:, sorting_indices]  # Reorder columns of accuracy matrix
    replacements = {
        "Counting & Probability": "Count.&Prob.",
        "Intermediate Algebra": "Inter. Algebra"
    }
    unique_subjects = [replacements.get(sub, sub) for sub in unique_subjects]  # Replace if in replacements
    # Step 3: Visualize the accuracy matrix using seaborn's heatmap
    sns.set(font_scale=1.25)
    plt.figure(figsize=(20, 8))
    # plt.figure(figsize=(12, 5))
    sns.heatmap(
        acc_by_subject,
        annot=True,
        fmt=".2f",
        cmap="PuBu",
        xticklabels=unique_subjects,
        yticklabels=methods,
        annot_kws={"size": 15}
    )

    plt.title(f"Accuracy Matrix per Subject and Attention Heads at Layer {pl} ({model_repo})", fontsize=14)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f'figures/accuracy_matrix_{model_repo}_{pl}.png', dpi=300)

    
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# Extract subject names from the dataset and sort them.
subjects = [sample['subject'] for sample in dataset]
unique_subjects = sorted(set(subjects))

# This dictionary will hold accuracy results per method per subject.
# The idea is to build a structure such that for each method (row label),
# for each subject (column), we collect a list of accuracy values (one per prune layer).
results = {}

# Loop over summaries and their corresponding prune_layers.
for summary, pl in zip(summaries, prune_layers):
    # Compute the accuracy matrix for this summary.
    acc_mat = get_accuracy_matrix(summary['head_accuracies'])
    
    # Define method labels.
    # (Assumes that summary['head_accuracies'] keys indicate the heads.
    #  We use all except the last key and then add "No Prune" as the final method.)
    num_heads = len(summary['head_accuracies'].keys())
    methods = [f"Prune Head {i}" for i in range(num_heads - 1)] + ["No Prune"]
    
    # Create and save a heat map for the current prune layer.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        acc_mat,
        annot=True,
        cmap="viridis",
        xticklabels=unique_subjects,
        yticklabels=methods
    )
    plt.title(f"Accuracy Heatmap for Prune Layer {pl}")
    plt.xlabel("Subject")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"heatmap_prune_layer_{pl}.png"))
    plt.close()
    
    # On the first iteration, initialize the results dictionary for each method.
    if not results:
        for m in methods:
            results[m] = {subject: [] for subject in unique_subjects}
            
    # Append the accuracy data from this summary for each method and subject.
    # (Assumes that the order of subjects in acc_mat columns corresponds to the order in unique_subjects.)
    for i, method in enumerate(methods):
        for j, subject in enumerate(unique_subjects):
            results[method][subject].append(acc_mat[i, j])

# --- Build the CSV file ---
# The CSV file will have these columns:
#  - First column: "Method" (which contains the strings like "Prune Head 0", etc.)
#  - For each subject (in order) and for each prune layer (in order), one column.
# For example, if you have subjects ['A', 'B'] and prune_layers [0.1, 0.2],
# then the headers (after the first column) will be:
#   "A (pl=0.1)", "A (pl=0.2)", "B (pl=0.1)", "B (pl=0.2)"
csv_columns = ["Method"]
for subject in unique_subjects:
    # Create one column header per prune_layers value.
    for pl in prune_layers:
        csv_columns.append(f"{subject} (pl={pl})")

# Build a list of rows (dictionaries) for the CSV.
rows = []
for method, subj_data in results.items():
    row = {"Method": method}
    for subject in unique_subjects:
        # There should be one accuracy value per prune layer, in the order the summaries were processed.
        for idx, pl in enumerate(prune_layers):
            row[f"{subject} (pl={pl})"] = subj_data[subject][idx]
    rows.append(row)

# Create the DataFrame and save it as CSV.
df = pd.DataFrame(rows, columns=csv_columns)
csv_filename = os.path.join(output_folder, "combined_results.csv")
df.to_csv(csv_filename, index=False)

print(f"CSV file saved to: {csv_filename}")

# breakpoint()
# acc_mat = get_accuracy_matrix(data['head_accuracies'])
# evaluate_best_of_n(acc_mat, n=16)