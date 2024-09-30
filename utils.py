from data_set import *
from results import *
import json
import os
def get_path_dataset(output_path,dataset_name,subdataset_name,train_length):
    if dataset_name == 'BBH':
        dataset = BBH_Dataset(f'chain-of-thought-hub/BBH/data/{subdataset_name}.json',train_length)
        filepath = f'results/{output_path}/{dataset_name}/{subdataset_name}/'
    elif dataset_name == 'MMLU':
        dataset = MMLU_Dataset(subdataset_name,train_length)
        filepath = f'results/{output_path}/{dataset_name}/{subdataset_name}/'
    elif dataset_name == 'emotion':
        train_length *= 28
        dataset = Emotion_Dataset(subdataset_name,train_length)
        filepath = f'results/{output_path}/{dataset_name}_28/{train_length}/'
    elif dataset_name == 'tacred':
        train_length *= 41
        dataset = TacRED_Dataset(subdataset_name,train_length)
        filepath = f'results/{output_path}/{dataset_name}_41/{train_length}/'
    else:
        print('unimplemented Dataset')
        exit(0)
    return dataset,filepath
def get_results(dataset_name,results):
    # results = None
    if dataset_name == 'BBH': 
        results = BBH_Results(results)
    elif dataset_name == 'MMLU':
        results = MMLU_Results(results)
    elif dataset_name == 'emotion':
        results = Emotion_Results(results)
    elif dataset_name == 'tacred':
        results = TacRED_Results(results)
    return results

def add_results(results,filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(f'{filepath}/config.json','r') as f:
            config = json.load(f)
    except:config = {}
    for k in results:
        config[k] = results[k]
    with open(f'{filepath}/config.json','w') as f:
        json.dump(config, f, indent=4)
