import pickle as pkl
from src.math500_utils import Math500Evaluator, load_dataset
import wandb
import os 
import json 
import re 
def extract_text_between(text, start_marker="user\n", end_marker="<|im_end|>"):
    """
    Extracts and returns the text between start_marker and end_marker.
    
    Args:
        text (str): The input string to search within.
        start_marker (str): The text marking the start of the extraction segment.
        end_marker (str): The text marking the end of the extraction segment.
    
    Returns:
        str: The extracted text. If no match is found, returns an empty string.
    """
    # Construct a pattern that escapes the markers, and use non-greedy match.
    pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return ""

# Initialize the evaluator and load the data.
evaluator = Math500Evaluator()
pkl_file = "tmp/best_of_16_with_prm_scores.pkl"
with open(pkl_file, 'rb') as f:
    data_ = pkl.load(f)

def get_file_from_wandb(file_name, wandb_run_id, root_dir="tmp"):
    api = wandb.Api()
    run = api.run(wandb_run_id) # wandb_run_id have format f"{entity}/{project}/{run_id}"
    run.file(file_name).download(replace=True, root=root_dir)
    with open(os.path.join(root_dir, file_name), "r") as f:
        data = json.load(f)
    return data

wandb_run_id = "hieunt91/our_method/7qpri4be"
data = get_file_from_wandb("wandb-summary.json", wandb_run_id=wandb_run_id)
test_texts = []
for text in data['dataset']['prompts']:
    match = extract_text_between(text)
    test_texts.append(match)

data = {
    'problem': [],
    'answer': [],
    'generated_answer': [],
    'prm_scores': []
}
for i in range(len(data_['problem'])):
    if data_['problem'][i] in test_texts:
        data['problem'].append(data_['problem'][i])
        data['answer'].append(data_['answer'][i])
        data['generated_answer'].append(data_['generated_answer'][i])
        data['prm_scores'].append(data_['prm_scores'][i])

print(len(data['problem']))
# A simple logging function so we can see evaluation results.
def log_results(name, per_subject_mean, per_subject_std):
    print(
        f"{name} |  Mean accuracy = {per_subject_mean*100:.2f}% (std = {per_subject_std*100:.2f}%)"
    )

# This function is provided to run evaluation given:
# • model_outputs: flattened candidates (grouped in blocks of num_return_sequences)
# • golden_answers: corresponding ground truth answers (repeated per candidate)
# • all_subjects: a list of subject labels (we use one subject 'all' here)
# • subject_set: the set of subjects to evaluate over (here, just ['all'])
# • evaluator: the evaluator instance
# • config_args: a dictionary containing "num_return_sequences" (i.e. best-of-N)
# • name: a label for the evaluation run (e.g., "Best-of-4")
def evaluate_model_outputs(model_outputs, golden_answers, evaluator, config_args, name):
    question_accs, per_subject_mean, per_subject_std = evaluator.get_success_rate(
        model_outputs,
        golden_answers,
        config_args["num_return_sequences"],
        stat_track=False
    )
    log_results(name, per_subject_mean, per_subject_std)
    return question_accs

# Define a helper function that takes the full data (with 16 responses per problem)
# and returns flattened candidate and gold answer lists for evaluation,
# where for each problem we select the top-n candidates based on their PRM scores.
def get_best_of_n(data, n):
    generated_answers = data["generated_answer"]
    gold_answers = data["answer"]
    prm_scores = data["prm_scores"]
    # There are 16 generations per problem.
    num_problems = len(generated_answers) // 16
    best_generated = []
    best_gold = []
    
    for i in range(num_problems):
        # Extract the block corresponding to one problem.
        block_generated = generated_answers[i*16:(i+1)*16][:n]
        block_gold = gold_answers[i*16:(i+1)*16][:n]
        block_scores = prm_scores[i*16:(i+1)*16][:n]
        
        sorted_candidates = sorted(zip(block_generated, block_scores), key=lambda x: x[1], reverse=True)
        top_candidate = sorted_candidates[0][0]
        best_generated.append(top_candidate)
        best_gold.append(block_gold[0])
    
    return best_generated, best_gold

def get_best_of_n_with_oracle(data, n, evaluator):
    generated_answers = data["generated_answer"]
    gold_answers = data["answer"]
    prm_scores = data["prm_scores"]
    # There are 16 generations per problem.
    num_problems = len(generated_answers) // 16
    best_generated = []
    best_gold = []
    
    for i in range(num_problems):
        # Extract the block corresponding to one problem.
        block_generated = generated_answers[i*16:(i+1)*16][:n]
        block_gold = gold_answers[i*16:(i+1)*16][:n]
        block_eval_scores, _, _ = evaluator.get_success_rate(
            block_generated,
            block_gold,
            1,
            stat_track=False
        )

        # block_scores = prm_scores[i*16:(i+1)*16][:n]
        
        sorted_candidates = sorted(zip(block_generated, block_eval_scores), key=lambda x: x[1], reverse=True)
        top_candidate = sorted_candidates[0][0]
        best_generated.append(top_candidate)
        best_gold.append(block_gold[0])
    
    return best_generated, best_gold

# =================================================================
# Now, perform evaluation for best-of-4, best-of-8, and best-of-16.
# =================================================================
import numpy as np 
ks = list(np.arange(16) + 1)

for k in ks:
    outputs_k, golds_k = get_best_of_n_with_oracle(data, k, evaluator)
    config_args_k = {"num_return_sequences": 1}
    accs_k = evaluate_model_outputs(
        outputs_k, golds_k, evaluator, config_args_k, f"Best-of-{k}"
    )

# outputs_4, golds_4 = get_best_of_n_with_oracle(data, 1, evaluator)
# config_args_4 = {"num_return_sequences": 1}
# print(len(outputs_4), len(golds_4))
# accs_4 = evaluate_model_outputs(
#     outputs_4, golds_4, evaluator, config_args_4, "Best-of-1")
    