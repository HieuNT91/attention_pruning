import json 
from .grading import grader 
from typing import Union, List, Dict
from collections import defaultdict
import numpy as np 
import os 

qwen_chat_math_prompt_template = """<|im_start|>system
Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant\n"""

llama_chat_math_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Please reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def check_and_download_math_500(dataset_path='datasets/math500/test.jsonl'):
    # download the dataset if it doesn't exist
    url = "https://media.githubusercontent.com/media/openai/prm800k/refs/heads/main/prm800k/math_splits/test.jsonl"
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        os.system(f"wget {url} -O {dataset_path}")
    else:
        print(f"Loaded dataset from {dataset_path}.")
        

def load_dataset(dataset_path):
    # check_and_download_math_500(dataset_path)
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def apply_prompt(sample, model_repo):
    if 'qwen' in model_repo.lower():
        if 'instruct' in model_repo.lower():
            prompt = qwen_chat_math_prompt_template.format(question=sample['problem'])
        else:
            prompt = f"Question: {sample['problem']}. Let's think step by step."
    elif 'lama' in model_repo.lower():
        if 'instruct' in model_repo.lower():
            prompt = llama_chat_math_prompt_template.format(question=sample['problem'])
        else:
            prompt = f"Question: {sample['problem']}. Let's think step by step."
    
    return prompt 

def extract_boxed_answer(llm_output):
    start = llm_output.find(r'\boxed{')
    if start == -1:
        return None  # No \boxed{} found

    # Start parsing after \boxed{
    i = start + len(r'\boxed{')
    stack = ['{']  # Start with one opening brace
    content = []

    # Loop through the string to extract content inside \boxed{}
    while i < len(llm_output):
        char = llm_output[i]

        # Add characters to content
        content.append(char)

        # Handle braces to ensure proper matching
        if char == '{':
            stack.append('{')
        elif char == '}':
            stack.pop()
            # If the stack is empty, we've matched all braces
            if not stack:
                break

        i += 1

    # Join the content and remove the last closing brace
    result = ''.join(content).strip()
    return result[:-1].strip() if result.endswith('}') else result.strip()

class Math500Evaluator:
    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {'mean': [], 'var': []})
    
    def get_stats(self):
        print('average success rate:', np.mean(self.stats['success_rate']['mean']))
        print('average success rate variance:', np.mean(self.stats['success_rate']['var']))
        return self.stats
    
    def get_success_rate(self, model_output, ground_truth_answers, num_return_sequences, stat_track=False):
        assert len(model_output) % num_return_sequences == 0, "Output size must be divisible by num_return_sequences."
        assert len(ground_truth_answers) == len(model_output) // num_return_sequences, f"Ground truth size ({len(ground_truth_answers)}) must match model output size ({len(model_output)}) divided by num_return_sequences ({num_return_sequences})."

        total_questions = len(ground_truth_answers)
        per_question_success_rates = []
        for i in range(total_questions):
            model_answers = model_output[i * num_return_sequences:(i + 1) * num_return_sequences]
            ground_truth_answer = ground_truth_answers[i]
            success_count = sum(
                grader.grade_answer(extract_boxed_answer(model_answer), ground_truth_answer) for model_answer in model_answers
            )
            per_question_success_rate = success_count / num_return_sequences
            per_question_success_rates.append(per_question_success_rate)

        if stat_track:
            self.stats['success_rate']['mean'].extend(per_question_success_rates)
            self.stats['success_rate']['var'].extend(per_question_success_rates)
            
        return per_question_success_rates, np.mean(per_question_success_rates), np.std(per_question_success_rates)