import pickle as pkl 
from src.math500_utils import load_dataset


data_0_2000 = pkl.load(open('shapley_prune/tmp/cache/Qwen2.5-Math-1.5B-Instruct_ns-1_s1_nrs16_sene30_bz3_start0_end2000/22_cached_outputs.pkl', 'rb'))
data_2000_3250 = pkl.load(open('/storage/hiu/project_2024/acl/DataDistillation/shapley_prune/tmp/home_cached_2000_3250.pkl', 'rb'))
data_3250_4500 = pkl.load(open('/storage/hiu/project_2024/acl/DataDistillation/shapley_prune/tmp/home_cached_3250_4000.pkl', 'rb'))

print(len(data_0_2000[1]))
print(len(data_2000_3250[1]))
print(len(data_3250_4500[1]))
full_outputs = data_0_2000[1] + data_2000_3250[1] + data_3250_4500[1]
print(len(full_outputs))
data = load_dataset('datasets/math500/test_updated.jsonl')


all_indices = []
data_ = data[0:2000]
subject_set = list(dict.fromkeys(x['subject'] for x in data_))
for chosen_subject in subject_set:
    # Filter data for the subject
    idx_list = [idx for idx, sample in enumerate(data[0:2000]) if sample['subject'] == chosen_subject]
    all_indices.extend(idx_list)
data_ = data[2000:3250]
subject_set = list(dict.fromkeys(x['subject'] for x in data_))
for chosen_subject in subject_set:
    # Filter data for the subject
    idx_list = [idx+2000 for idx, sample in enumerate(data[2000:3250]) if sample['subject'] == chosen_subject]
    all_indices.extend(idx_list)
data_ = data[3250:4500]
subject_set = list(dict.fromkeys(x['subject'] for x in data_))
for chosen_subject in subject_set:
    # Filter data for the subject
    idx_list = [idx+3250 for idx, sample in enumerate(data[3250:4500]) if sample['subject'] == chosen_subject]
    all_indices.extend(idx_list) 

reordered_data = [data[i] for i in all_indices]

print(len(reordered_data))
print(reordered_data[0].keys())

duplicated_data = [entry['problem'] for entry in reordered_data for _ in range(16)]
duplicated_answer = [entry['answer'] for entry in reordered_data for _ in range(16)]

saved_data = {
    "problem": duplicated_data,
    "answer": duplicated_answer,
    "generated_answer": full_outputs
}
print(duplicated_data[0])
print(full_outputs[0])
print(len(saved_data['problem']))
print(len(saved_data['answer']))
print(len(saved_data['generated_answer']))

# save saved_data
pkl.dump(saved_data, open('shapley_prune/tmp/best_of_16.pkl', 'wb'))
breakpoint()
