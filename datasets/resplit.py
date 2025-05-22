import os
import json
import random

def check_and_download_math_500(dataset_path='datasets/math500/test.jsonl'):
    # Download the dataset if it doesn't exist
    url = "https://media.githubusercontent.com/media/openai/prm800k/refs/heads/main/prm800k/math_splits/test.jsonl"
    if not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        os.system(f"wget {url} -O {dataset_path}")
    else:
        print(f"Loaded dataset from {dataset_path}.")

def load_dataset(dataset_path):
    # Ensure the dataset exists, then load it
    check_and_download_math_500(dataset_path)
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def partition_train_data(train_data, num_samples=4000):
    """
    Partition the train dataset into two parts:
    1. `selected_samples`: Randomly selected `num_samples` to be added to the test dataset.
    2. `remaining_train_data`: The remaining train dataset after removing the selected samples.
    """
    if len(train_data) < num_samples:
        print(f"Train dataset has only {len(train_data)} samples. All samples will be used.")
        return train_data, []  # All train data becomes selected, leaving no remaining train data.
    
    # Randomly select `num_samples` without replacement
    selected_samples = random.sample(train_data, num_samples)
    
    # Create a set of IDs (or hashes) of selected samples for efficient removal
    selected_ids = set(id(item) for item in selected_samples)
    
    # Filter out selected samples to get the remaining train data
    remaining_train_data = [item for item in train_data if id(item) not in selected_ids]
    
    return selected_samples, remaining_train_data

# Paths to the train and test datasets
train_dataset_path = 'math500/train.jsonl'
test_dataset_path = 'math500/test.jsonl'

# Load train and test datasets
train_data = load_dataset(train_dataset_path)
test_data = load_dataset(test_dataset_path)

# Partition train data into random samples and remaining train data
selected_samples, remaining_train_data = partition_train_data(train_data, num_samples=4000)

# Add the selected samples to the test dataset
test_data.extend(selected_samples)

# Save the updated test dataset
updated_test_dataset_path = 'math500/test_updated.jsonl'
with open(updated_test_dataset_path, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"Updated test dataset saved to {updated_test_dataset_path}.")
print(f"Test dataset size after addition: {len(test_data)}")

# Save the remaining train dataset
updated_train_dataset_path = 'math500/train_updated.jsonl'
with open(updated_train_dataset_path, 'w', encoding='utf-8') as f:
    for item in remaining_train_data:
        f.write(json.dumps(item) + '\n')

print(f"Updated train dataset (remaining samples) saved to {updated_train_dataset_path}.")
print(f"Remaining train dataset size: {len(remaining_train_data)}")