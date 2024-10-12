import argparse
from data_set import *
from utils import *
from model_arithmetic import ModelArithmetic, PromptedLLM
from functools import partial
import nevergrad as ng
import torch.nn as nn
import torch
import re
from prompt import *
from datetime import datetime
start_time = datetime.now()
print(start_time)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = "meta-llama/Meta-Llama-3.1-8B")
parser.add_argument('--output_path', type=str, default = "results")
parser.add_argument('--dataset_name', type=str, default = 'BBH')
parser.add_argument('--train_length', type=int, default = 16)
parser.add_argument('--binary', action='store_true', help="binary mode")

loss_fn = nn.CrossEntropyLoss()
args = parser.parse_args()
model_name = args.model
output_path = args.output_path
dataset_name = args.dataset_name
train_length = args.train_length

max_length = 10

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
else : TASKS = [1]
for subdataset_name in TASKS:
    output_path = args.output_path
    if args.binary:
        output_path+='_01'
        print('start binary mode')
    dataset,filepath = get_path_dataset(output_path,dataset_name,subdataset_name,train_length)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sub = train_length//2
    dataset.show_example()

    if os.path.exists(f'{filepath}/non_reweight_baseline.json'):
        print(f'{filepath}/confinon_reweight_baseline.json')
        continue

    emotion_candidates = ["admiration",
                    "amusement",
                    "anger",
                    "annoyance",
                    "approval",
                    "caring",
                    "confusion",
                    "curiosity",
                    "desire",
                    "disappointment",
                    "disapproval",
                    "disgust",
                    "embarrassment",
                    "excitement",
                    "fear",
                    "gratitude",
                    "grief",
                    "joy",
                    "love",
                    "nervousness",
                    "optimism",
                    "pride",
                    "realization",
                    "relief",
                    "remorse",
                    "sadness",
                    "surprise",
                    "neutral"]

    def replace_first_from_list(text, replacements, new_substring):
        pattern = r'|'.join(map(re.escape, replacements))
        result = re.sub(pattern, new_substring, text, count=1)
        return result

    def split_list_into_equal_parts(my_list, size=16, max_len = None):
        if max_len == None: max_len = len(my_list)
        return [my_list[i:i + size] for i in range(0, max_len, size)]

    def replace_last_number(s, replacement):
        match = re.findall(r'-?\d+\.?\d*', s)
        if match:
            last_number = match[-1]
            s = re.sub(r'-?\d+\.?\d*$', replacement, s[::-1], count=1)[::-1]
        return s
    class Merged_model:
        def __init__(self, model_path, prompt_template, prompt, max_length, train_data, number_of_shot, weights, normalized = True):
            random.shuffle(train_data)
            self.train_data = split_list_into_equal_parts(train_data, number_of_shot)
            weights = list(weights)
            if normalized:
                sum_num = sum(weights)
                if sum_num!=0:
                    for i in range(len(weights)):
                        weights[i] = weights[i]/sum_num
            assert len(weights)==len(self.train_data)
            llms = [j * PromptedLLM(prompt(i,subdataset_name), prompt_template=prompt_template) for i,j in zip(self.train_data,weights)]
            formula = sum(llms)
            self.model = ModelArithmetic(formula, default_model=model_path)
            self.max_length = max_length

        def generate(self, test_data):
            inputs = [i['input'] for i in test_data]
            truths = [i['output'] for i in test_data]
            results = []
            for i,j in tzip(inputs, truths):
                outputs = self.model.generate_text(i, max_length=self.max_length, top_k = 1)
                results.append({'input':i, 'output':outputs[0], 'truth':j})
            return results

        def compute_loss(self, inputs, truth):
            outputs = self.model.generate_text(inputs, max_length=max_length, top_k = 1)
            if dataset_name == 'BBH':
                truth[0] = re.sub(r'\([^\)]+\)', truth[0], outputs[0])
            if dataset_name == 'MMLU':
                truth[0] = re.sub(r'[A-Z]', truth[0], outputs[0], count = 1)
            if dataset_name == 'emotion':
                truth[0] = replace_first_from_list(outputs[0],emotion_candidates,truth[0])
            truth = self.model.tokenizer(truth, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
            logits = self.model.tmp
            tensor_list = [torch.stack([v for k, v in sorted(d.items())], dim=0) for d in logits]
            logits = torch.stack(tensor_list, dim=0)
            a = torch.argmax(logits, dim=-1)
            if a[0][0] == truth[0][0]:
                min_length = min(logits.size(1), truth.size(1))
                logits = logits[:, :min_length, :]
                truth = truth[:, :min_length]
            else:            
                min_length = min(logits.size(1), truth.size(1)-1)
                logits = logits[:, :min_length, :]
                truth = truth[:, 1:min_length+1]
            loss = loss_fn(logits.view(-1, logits.size(-1)), truth.view(-1))
            return loss.item()

        def calculate_loss(self, test_data):
            inputs = [i['input'] for i in test_data]
            truths = [i['output'] for i in test_data]
            loss = 0
            for i,j in tzip(inputs, truths):
                loss += self.compute_loss([i],[' '+j])
            return loss

    tmp_results=[]
    def get_score(weights,model_path, prompt_template, prompt, max_length, train_data, number_of_shot,test_data):
        if sum(weights)==0:return 999999999
        score = 0
        g =int(sub/number_of_shot)
        tmp = list(weights[:g])
        model = Merged_model(model_path, prompt_template, prompt, max_length, train_data, number_of_shot,tmp)
        score +=model.calculate_loss(test_data)
        tmp = weights[g:]
        model = Merged_model(model_path, prompt_template, prompt, max_length, test_data, number_of_shot,tmp)
        score +=model.calculate_loss(train_data)
        return score
    tmp={}
    for number_of_shot in [2,4,8]:
        if dataset_name == 'emotion': number_of_shot *= 14
        init_pas = [1/(train_length//number_of_shot)] * (train_length//number_of_shot)

        model = Merged_model(model_name, prompt_templates[dataset_name], prompts[dataset_name], max_length, dataset.train, number_of_shot, init_pas)
        results = model.generate(dataset.test)
        results = get_results(dataset_name, results)

        results.save(f'{filepath}/non_reweight_baseline.json')
        our_score = results.calculate_score()
        tmp[number_of_shot] = our_score

    end_time = datetime.now()
    print(end_time)
    tmp['start_time'] = str(start_time)
    tmp['end_time'] = str(end_time)
    add_results(tmp,filepath)

        
