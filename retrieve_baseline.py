import argparse
from data_set import *
from prompt import *
from results import *
import numpy as np
from model_arithmetic import ModelArithmetic, PromptedLLM
import warnings
from sentence_transformers import SentenceTransformer
from datetime import datetime
from utils import *
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = "NousResearch/Llama-2-7b-hf")
parser.add_argument('--output_path', type=str, default = "results")
parser.add_argument('--dataset_name', type=str, default = 'BBH')
parser.add_argument('--train_length', type=int, default = 32)
args = parser.parse_args()
model_name = args.model
output_path = args.output_path
dataset_name = args.dataset_name
max_length = 10

def evaluate(train_data,test_data):
    results = []
    for i,j in tzip(train_data,test_data):
        model = PromptedLLM(prompt(i,subdataset_name), prompt_template=prompt_template)
        formula = model
        model = ModelArithmetic(formula, default_model=model_name)
        outputs = model.generate_text(j['input'], max_length=max_length, top_k = 1)
        results.append({'input':j['input'], 'output':outputs[0], 'truth':j['output']})
    return results

if dataset_name == 'BBH':
    TASKS = [
        'temporal_sequences',
        'disambiguation_qa',
        'date_understanding',
        'tracking_shuffled_objects_three_objects',
        'penguins_in_a_table',
        'geometric_shapes',
        'snarks',
        'ruin_names',
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_five_objects',
        'logical_deduction_three_objects',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'movie_recommendation',
        'salient_translation_error_detection',
        'reasoning_about_colored_objects'
    ]
elif dataset_name == 'MMLU':
    TASKS  = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
    ]
else: TASKS = [1]

train_length = args.train_length
for subdataset_name in TASKS:
    start_time = datetime.now()
    print(start_time)
    print(subdataset_name)
    dataset,filepath = get_path_dataset(output_path,dataset_name,subdataset_name,train_length)
    if os.path.exists(f'{filepath}/re_config.json'):
        print(f'{filepath}/re_config.json')

    model = SentenceTransformer('all-distilroberta-v1')
    inputs_test = [i['input'] for i in dataset.test]
    inputs_train = [i['input'] for i in dataset.train]
    embeddings1 = model.encode(inputs_test)
    embeddings2 = model.encode(inputs_train)
    cosine_similarities = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1, axis=1)[:, np.newaxis] * np.linalg.norm(embeddings2, axis=1))
    prompt_template=prompt_templates[dataset_name]
    prompt=prompts[dataset_name]
    all_results = {}
    for n in [2,4,8]:
        if n > train_length:continue
        top_k_demo = []
        for idx, similarities in enumerate(cosine_similarities):
            top_n_indices = np.argsort(similarities)[::-1][:n]
            top_k_demo.append([dataset.train[top_idx] for top_idx in top_n_indices])

        results = evaluate(top_k_demo, dataset.test)
        results = get_results(dataset_name, results)
        all_results[n] = results.calculate_score()

    end_time = datetime.now()
    print(end_time)
    all_results['start_time'] = str(start_time)
    all_results['end_time'] = str(end_time)
    add_results({'retrieve':all_results},filepath)

        
