import os
import json
from abc import ABC, abstractmethod
from tqdm.contrib import tzip
import re

class Results:
    def __init__(self, results):
        if isinstance(results, list):
            self.results = results
        elif isinstance(results, str):
            with open(results, 'r') as file:
                self.results = json.load(file)
    
    def post_processing(self):
        pass

    @abstractmethod
    def calculate_score(self):
        pass

    def save(self, path):
        self.post_processing()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

class BBH_Results(Results):
    def post_processing(self):
        candidates = ['(e)', '(i)', '(l)', '(k)', '(a)', '(b)', '(g)', '(f)', '(o)', '(d)', '(q)', '(n)', '(h)', '(j)', '(p)', '(c)', '(r)', '(m)']
        for i in range(len(self.results)):
            truth = self.results[i]['truth'].lower()
            output = self.results[i]['output'].lower()
            try:
                output = output.split('question')[0]
            except:pass
            answer_output = None
            for answer in candidates:
                if output in answer or answer in output:
                    answer_output = answer
                    break
            try:
                self.results[i]['answer'] = answer_output.upper()
            except:self.results[i]['answer'] = ''

    def calculate_score(self):
        self.post_processing()
        count = 0
        for result in self.results:
            if result['truth']==result['answer']:count+=1
        return count/len(self.results)

class MMLU_Results(Results):
    def post_processing(self):
        candidates = ['A','B','C','D']
        for i in range(len(self.results)):
            truth = self.results[i]['truth']
            output = self.results[i]['output']
            try:
                output = output.split('Question')[0]
            except:pass
            answer_output = None
            for answer in candidates:
                if output in answer or answer in output:
                    answer_output = answer
                    break
            try:
                self.results[i]['answer'] = answer_output.upper()
            except:self.results[i]['answer'] = ''

    def calculate_score(self):
        self.post_processing()
        count = 0
        for result in self.results:
            if result['truth']==result['answer']:count+=1
        return count/len(self.results)

class Emotion_Results(Results):
    def post_processing(self):
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
        for i in range(len(self.results)):
            truth = self.results[i]['truth'].lower()
            output = self.results[i]['output'].lower()
            try:
                output = output.split('comment')[0]
            except:pass
            answer_output = None
            for answer in candidates:
                if output in answer or answer in output:
                    answer_output = answer
                    # break
            try:
                self.results[i]['answer'] = answer_output.lower()
            except:self.results[i]['answer'] = ''

    def calculate_score(self):
        self.post_processing()
        count = 0
        for result in self.results:
            if result['truth']==result['answer']:count+=1
        return count/len(self.results)

class TacRED_Results(Results):
    def post_processing(self):
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
        for i in range(len(self.results)):
            truth = self.results[i]['truth'].lower()
            output = self.results[i]['output'].lower()
            try:
                output = output.split('sentence:')[0]
            except:pass
            answer_output = None
            for answer in candidates:
                if output in answer or answer in output:
                    answer_output = answer
                    # break
            try:
                self.results[i]['answer'] = answer_output.lower()
            except:self.results[i]['answer'] = ''

    def calculate_score(self):
        self.post_processing()
        count = 0
        for result in self.results:
            if result['truth']==result['answer']:count+=1
        return count/len(self.results)
    
