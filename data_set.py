import json
import random
import pandas as pd
import os
random.seed(77)
import jsonlines
from collections import defaultdict
from datasets import load_dataset

class Dataset:
    def __init__(self):
        self.train = []
        self.test = []

    def show_example(self):
        print(f'train example: {self.train[0]}')
        print(f'test example: {self.test[0]}')
        print(f'train example: {self.train[1]}')
        print(f'test example: {self.test[1]}')

class BBH_Dataset(Dataset):
    def __init__(self,path,train_length = 64):
        super().__init__()
        with open(path,'r') as f:
            datas = json.load(f)
            test_data = datas['examples'][train_length:]
            train_data = datas['examples'][:train_length][:train_length]
            random.shuffle(train_data)
        self.train = [{'input':data['input'],'output':data['target']} for data in train_data]
        self.test = [{'input':data['input'],'output':data['target']} for data in test_data]

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt, df.iloc[idx, k + 1]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


choices = ["A", "B", "C", "D"]
class MMLU_Dataset(Dataset):

    def __init__(self,subdataset_name,train_length = 64):
        super().__init__()
        dev_df = pd.read_csv(os.path.join('datasets/MMLU/data', "test", subdataset_name + "_test.csv"), header=None)[:64][:train_length]
        # chunked_dfs = [dev_df.iloc[i:i + num_rows_per_chunk] for i in range(0, len(dev_df), num_rows_per_chunk)]
        test_df = pd.read_csv(os.path.join('datasets/MMLU/data', "test", subdataset_name + "_test.csv"), header=None)[64:][:200]
        self.train = []
        for i in range(len(dev_df)):
            prompt, answer = format_example(dev_df,i,False)
            self.train.append({'input':prompt,'output':answer})
        self.test = []
        for i in range(len(test_df)):
            prompt, answer = format_example(test_df,i,False)
            self.test.append({'input':prompt,'output':answer})

class Emotion_Dataset(Dataset):
    def __init__(self,subdataset_name = None,train_length = 64):
        assert train_length % 28 ==0
        candidates = ["admiration",
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
        super().__init__()
        self.train = []
        self.test = []
        with open('datasets/emotion/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('datasets/emotion/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//28):
            random.shuffle(candidates)
            for candidate in candidates:
                self.train.append(random.choice(grouped_data[candidate]))
        assert len(self.train) == train_length

class TacRED_Dataset(Dataset):
    def __init__(self,subdataset_name = None,train_length = 128):
        assert train_length % 41 ==0
        candidates = [
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
        super().__init__()
        self.train = []
        self.test = []
        with open('datasets/tacred/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('datasets/tacred/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//41):
            random.shuffle(candidates)
            for candidate in candidates:
                self.train.append(random.choice(grouped_data[candidate]))
        assert len(self.train) == train_length
         
  
if __name__ == "__main__":
    dataset = Webqa_Dataset()
    dataset.show_example()

