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
        dev_df = pd.read_csv(os.path.join('chain-of-thought-hub/MMLU/data', "test", subdataset_name + "_test.csv"), header=None)[:64][:train_length]
        # chunked_dfs = [dev_df.iloc[i:i + num_rows_per_chunk] for i in range(0, len(dev_df), num_rows_per_chunk)]
        test_df = pd.read_csv(os.path.join('chain-of-thought-hub/MMLU/data', "test", subdataset_name + "_test.csv"), header=None)[64:][:200]
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
        with open('language-model-arithmetic/datasets/emotion/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('language-model-arithmetic/datasets/emotion/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//28):
            for candidate in candidates:
                self.train.append(random.choice(grouped_data[candidate]))
        assert len(self.train) == train_length

class Banking77_Dataset(Dataset):
    def __init__(self,subdataset_name = None,train_length = 128):
        assert train_length % 77 ==0
        candidates = [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment?",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal"
        ]
        super().__init__()
        self.train = []
        self.test = []
        with open('language-model-arithmetic/datasets/banking77/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['output'] = candidates[sample['output']] # convert from number to label
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('language-model-arithmetic/datasets/banking77/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['output'] = candidates[sample['output']] # convert from number to label
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//77):
            for candidate in candidates:
                self.train.append(random.choice(grouped_data[candidate]))
        assert len(self.train) == train_length

class Discovery_Dataset(Dataset):
    def __init__(self,subdataset_name = None,train_length = 128):
        assert train_length % 174 ==0
        candidates = [
            "[no-conn]",
            "absolutely,",
            "accordingly",
            "actually,",
            "additionally",
            "admittedly,",
            "afterward",
            "again,",
            "already,",
            "also,",
            "alternately,",
            "alternatively",
            "although,",
            "altogether,",
            "amazingly,",
            "and",
            "anyway,",
            "apparently,",
            "arguably,",
            "as_a_result,",
            "basically,",
            "because_of_that",
            "because_of_this",
            "besides,",
            "but",
            "by_comparison,",
            "by_contrast,",
            "by_doing_this,",
            "by_then",
            "certainly,",
            "clearly,",
            "coincidentally,",
            "collectively,",
            "consequently",
            "conversely",
            "curiously,",
            "currently,",
            "elsewhere,",
            "especially,",
            "essentially,",
            "eventually,",
            "evidently,",
            "finally,",
            "first,",
            "firstly,",
            "for_example",
            "for_instance",
            "fortunately,",
            "frankly,",
            "frequently,",
            "further,",
            "furthermore",
            "generally,",
            "gradually,",
            "happily,",
            "hence,",
            "here,",
            "historically,",
            "honestly,",
            "hopefully,",
            "however",
            "ideally,",
            "immediately,",
            "importantly,",
            "in_contrast,",
            "in_fact,",
            "in_other_words",
            "in_particular,",
            "in_short,",
            "in_sum,",
            "in_the_end,",
            "in_the_meantime,",
            "in_turn,",
            "incidentally,",
            "increasingly,",
            "indeed,",
            "inevitably,",
            "initially,",
            "instead,",
            "interestingly,",
            "ironically,",
            "lastly,",
            "lately,",
            "later,",
            "likewise,",
            "locally,",
            "luckily,",
            "maybe,",
            "meaning,",
            "meantime,",
            "meanwhile,",
            "moreover",
            "mostly,",
            "namely,",
            "nationally,",
            "naturally,",
            "nevertheless",
            "next,",
            "nonetheless",
            "normally,",
            "notably,",
            "now,",
            "obviously,",
            "occasionally,",
            "oddly,",
            "often,",
            "on_the_contrary,",
            "on_the_other_hand",
            "once,",
            "only,",
            "optionally,",
            "or,",
            "originally,",
            "otherwise,",
            "overall,",
            "particularly,",
            "perhaps,",
            "personally,",
            "plus,",
            "preferably,",
            "presently,",
            "presumably,",
            "previously,",
            "probably,",
            "rather,",
            "realistically,",
            "really,",
            "recently,",
            "regardless,",
            "remarkably,",
            "sadly,",
            "second,",
            "secondly,",
            "separately,",
            "seriously,",
            "significantly,",
            "similarly,",
            "simultaneously",
            "slowly,",
            "so,",
            "sometimes,",
            "soon,",
            "specifically,",
            "still,",
            "strangely,",
            "subsequently,",
            "suddenly,",
            "supposedly,",
            "surely,",
            "surprisingly,",
            "technically,",
            "thankfully,",
            "then,",
            "theoretically,",
            "thereafter,",
            "thereby,",
            "therefore",
            "third,",
            "thirdly,",
            "this,",
            "though,",
            "thus,",
            "together,",
            "traditionally,",
            "truly,",
            "truthfully,",
            "typically,",
            "ultimately,",
            "undoubtedly,",
            "unfortunately,",
            "unsurprisingly,",
            "usually,",
            "well,",
            "yet,",
        ]
        super().__init__()
        self.train = []
        self.test = []
        with open('language-model-arithmetic/datasets/discovery/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['output'] = candidates[sample['output']] # convert from number to label
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('language-model-arithmetic/datasets/discovery/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                sample['output'] = candidates[sample['output']] # convert from number to label
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//174):
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
        with open('language-model-arithmetic/datasets/tacred/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.test.append(sample)
        random.shuffle(self.test)
        self.test = self.test[:500]
        grouped_data = defaultdict(list)
        with open('language-model-arithmetic/datasets/tacred/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                grouped_data[sample['output']].append(sample)
        for _ in range(train_length//41):
            for candidate in candidates:
                self.train.append(random.choice(grouped_data[candidate]))
        assert len(self.train) == train_length

class Gsm8k_Dataset(Dataset):
    def __init__(self,subdataset_name = None,train_length = 64):
        super().__init__()
        self.train = []
        self.test = []
        with open('language-model-arithmetic/datasets/gsm8k/test.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.test.append({'input':sample['question'],'output':sample['answer']})
        random.shuffle(self.test)
        self.test = self.test[:500]
        with open('language-model-arithmetic/datasets/gsm8k/train.jsonl','r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.train.append({'input':sample['question'],'output':sample['answer']})
        self.train = self.train[:train_length]


class Webqa_Dataset(Dataset):
    def __init__(self, subdataset_name=None, train_length=64):
        super().__init__()
        self.train = []
        self.test = []
        ds = load_dataset("Stanford/web_questions")
        train_sample = ds['train'].shuffle(seed=77).select(range(train_length))
        test_sample = ds['test'].shuffle(seed=77).select(range(500))
        for i in train_sample:
            self.train.append({'input':i['question'],'output':i['answers'][0]})
        for i in test_sample:
            self.test.append({'input':i['question'],'output':i['answers'][0]})
                

class Nl2bash_Dataset(Dataset):
    def __init__(self, subdataset_name=None, train_length = 64):
        super().__init__()
        self.train = []
        self.test = []
        ds = load_dataset("jiacheng-ye/nl2bash")
        train_sample = ds['train'].shuffle(seed=77).select(range(train_length))
        test_sample = ds['test'].shuffle(seed=77).select(range(500))
        for i in train_sample:
            self.train.append({'input':i['nl'],'output':i['bash']})
        for i in test_sample:
            self.test.append({'input':i['nl'],'output':i['bash']})
             

class Geoquery_Dataset(Dataset):
    def __init__(self, subdataset_name=None, train_length = 64):
        super().__init__()
        self.train = []
        self.test = []
        ds = load_dataset("vaishali/geoQuery-tableQA")
        train_sample = ds['train'].shuffle(seed=77).select(range(train_length))
        test_sample = ds['test'].shuffle(seed=77)
        for i in train_sample:
            self.train.append({'input':i['question'],'output':i['query']})
        for i in test_sample:
            self.test.append({'input':i['question'],'output':i['query']})
       
if __name__ == "__main__":
    dataset = Webqa_Dataset()
    dataset.show_example()

