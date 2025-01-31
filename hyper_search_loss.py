import argparse
from data_set import *
from prompt import *
from results import *
from utils import *
from model_arithmetic import ModelArithmetic, PromptedLLM
from functools import partial
import nevergrad as ng
import torch.nn as nn
import torch
import re
import random
from datetime import datetime
import pdb
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = "mistralai/Mistral-7B-v0.3")
parser.add_argument('--output_path', type=str, default = "results")
parser.add_argument('--dataset_name', type=str, default = 'BBH')
parser.add_argument('--subdataset_name', type=str, default = 'date_understanding')
parser.add_argument('--train_length', type=int, default = 16)
parser.add_argument('--binary', action='store_true', help="binary mode")
start_time = datetime.now()
print(start_time)
loss_fn = nn.CrossEntropyLoss()
args = parser.parse_args()
model_name = args.model
output_path = args.output_path
dataset_name = args.dataset_name
subdataset_name = args.subdataset_name
train_length = args.train_length
max_length = 10
if args.binary:
    output_path+='_01'
    print('start binary mode')
dataset,filepath = get_path_dataset(output_path,dataset_name,subdataset_name,train_length)
os.makedirs(os.path.dirname(filepath), exist_ok=True)

dataset.show_example()
if dataset_name == 'emotion': train_length *= 28
if dataset_name == 'tacred': train_length *= 41
sub = train_length//2
if os.path.exists(f'{filepath}/config.json'):
    print(f'{filepath}/config.json')
    exit()

emotion_candidates = [
                "disapproval",
                "admiration",
                "amusement",
                "anger",
                "annoyance",
                "approval",
                "caring",
                "confusion",
                "curiosity",
                "desire",
                "disappointment",
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

tacred_candidates = [
        'org:founded_by', 
        'per:employee_of', 
        'org:alternate_names', 
        'per:cities_of_residence', 
        'per:children', 
        'per:title', 
        'per:siblings', 
        'per:religion', 
        'per:age', 
        'org:website', 
        'per:stateorprovinces_of_residence', 
        'org:member_of', 
        'org:top_members/employees', 
        'per:countries_of_residence', 
        'org:city_of_headquarters', 
        'org:members', 
        'org:country_of_headquarters', 
        'per:spouse', 
        'org:stateorprovince_of_headquarters', 
        'org:number_of_employees/members', 
        'org:parents', 
        'org:subsidiaries', 
        'per:origin', 
        'org:political/religious_affiliation', 
        'per:other_family', 
        'per:stateorprovince_of_birth', 
        'org:dissolved', 
        'per:date_of_death', 
        'org:shareholders', 
        'per:alternate_names', 
        'per:parents', 
        'per:schools_attended', 
        'per:cause_of_death', 
        'per:city_of_death', 
        'per:stateorprovince_of_death', 
        'org:founded', 
        'per:country_of_birth', 
        'per:date_of_birth', 
        'per:city_of_birth', 
        'per:charges', 
        'per:country_of_death'
    ]

def replace_first_from_list(text, replacements, new_substring):
    pattern = r'|'.join(map(re.escape, replacements))
    result = re.sub(pattern, new_substring, text, count=1)
    return result

def split_list_into_equal_parts(my_list, size=16, max_len=None):
    if max_len is None: max_len = len(my_list)
    parts = [my_list[i:i + size] for i in range(0, max_len, size)]
    if len(parts) > 1 and len(parts[-1]) < size:
        parts[-2].extend(parts[-1])
        parts = parts[:-1]
    return parts

def replace_last_number(s, replacement):
    match = re.findall(r'-?\d+\.?\d*', s)
    if match:
        last_number = match[-1]
        s = re.sub(r'-?\d+\.?\d*$', replacement, s[::-1], count=1)[::-1]
    return s
class Merged_model:
    def __init__(self, model_path, prompt_template, prompt, max_length, train_data, number_of_shot, weights, normalized = True):
        self.train_data = split_list_into_equal_parts(train_data, number_of_shot)
        weights = list(weights)
        if normalized:
            sum_num = sum(weights)
            if sum_num!=0:
                for i in range(len(weights)):
                    weights[i] = weights[i]/sum_num
        assert len(weights)==len(self.train_data)
        llms = [j * PromptedLLM(prompt(i,subdataset_name), prompt_template=prompt_template) for i,j in zip(self.train_data,weights) if j != 0]
        formula = sum(llms)
        self.model = ModelArithmetic(formula, default_model=model_path)
        self.max_length = max_length

    def generate(self, test_data):
        print('Start inference')
        inputs = [i['input'] for i in test_data]
        truths = [i['output'] for i in test_data]
        results = []
        for i,j in tzip(inputs, truths):
            outputs = self.model.generate_text(i, max_length=self.max_length, top_k = 1)
            results.append({'input':i, 'output':outputs[0], 'truth':j})
        return results

    def compute_loss(self, inputs, truth):
        outputs = self.model.generate_text(inputs, max_length = self.max_length, top_k = 1)
        if dataset_name == 'BBH':
            truth[0] = re.sub(r'\([^\)]+\)', truth[0].strip(), outputs[0])
        if dataset_name == 'MMLU':
            truth[0] = re.sub(r'[A-Z]', truth[0].strip(), outputs[0], count = 1)
        if dataset_name == 'emotion':
            truth[0] = replace_first_from_list(outputs[0],emotion_candidates,truth[0].strip())
        if dataset_name == 'tacred':
            truth[0] = replace_first_from_list(outputs[0],tacred_candidates,truth[0].strip())
        if 'gsm8k' in dataset_name:
            truth[0] = re.sub(r'(-?\d+\.?\d*)(?!.*(-?\d+\.?\d*))', truth[0], outputs[0])
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
        for i,j in zip(inputs, truths):
            loss += self.compute_loss([i],[' '+j])
        return loss

tmp_results=[]
def get_score(weights,model_path, prompt_template, prompt, max_length, train_data, number_of_shot,test_data):
    if sum(weights)==0:return 999999999
    score = 0
    g =int(sub/number_of_shot)
    tmp = list(weights[:g])
    if sum(tmp) == 0:return 999999999
    model = Merged_model(model_path, prompt_template, prompt, max_length, train_data, number_of_shot,tmp)
    score += model.calculate_loss(test_data)
    tmp = weights[g:]
    if sum(tmp) == 0:return 999999999
    model = Merged_model(model_path, prompt_template, prompt, max_length, test_data, number_of_shot,tmp)
    score += model.calculate_loss(train_data)
    return score

tmp = {}
for number_of_shot in [2,4,8]:
    if dataset_name in ['emotion', 'tacred']: number_of_shot = train_length // number_of_shot //2
    if number_of_shot >= sub:continue
    get_score_partial = partial(get_score, 
                                model_path=model_name, 
                                prompt_template=prompt_templates[dataset_name],
                                prompt=prompts[dataset_name],
                                max_length=max_length,
                                train_data=dataset.train[:sub], 
                                number_of_shot=number_of_shot,
                                test_data = dataset.train[sub:])
    if args.binary:
        instrum = ng.p.Choice([0,1],repetitions=train_length//number_of_shot)
    else:
        init_pas = [1/(train_length//number_of_shot)] * (train_length//number_of_shot)
        instrum = ng.p.Array(
            init=init_pas,
            upper=[1.5] * (train_length//number_of_shot),
            lower=[-1.5] * (train_length//number_of_shot),
        )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=20)
    # recommendation = optimizer.ask()
    with tqdm(total=optimizer.budget) as pbar:
        for i in range(optimizer.budget):
            recommendation = optimizer.ask()
            loss = get_score_partial(recommendation.value)
            optimizer.tell(recommendation, loss) 
            recommendation = optimizer.ask()
            pbar.update(1)
    # recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    score = get_score_partial(recommendation.value)
    weights = list(recommendation.value)
    tmp[number_of_shot] = (score,weights)
sorted_items = sorted(tmp.items(), key=lambda item: item[1])
print(sorted_items)
final_weights = sorted_items[0]

model = Merged_model(model_name, prompt_templates[dataset_name], prompts[dataset_name], max_length, dataset.train, final_weights[0], final_weights[1][1])
results = model.generate(dataset.test)
results = get_results(dataset_name, results)

results.save(f'{filepath}/our_method.json')
our_score = results.calculate_score()


model = Merged_model(model_name, prompt_templates[dataset_name], prompts[dataset_name], max_length, dataset.train, train_length, [1])
results = model.generate(dataset.test)
results = get_results(dataset_name, results)
results.save(f'{filepath}/baseline.json')
baseline_score = results.calculate_score()

end_time = datetime.now()
print(end_time)

add_results({'lamda':json.dumps(final_weights[1][1]),'number_of_shot':final_weights[0],'our_score':our_score,'baseline_score':baseline_score,'start_time':str(start_time),'end_time':str(end_time)},filepath)

