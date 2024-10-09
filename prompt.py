
def get_BBH_prompt(datas, subject=None):
    ans = ''
    for data in datas:
        ans += f"Question: {data['input']}\nAnswer: {data['output']}\n\n"
    return ans

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def gen_MMLU_prompt(datas, subject):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    for data in datas:
        prompt += f"Question: {data['input']} {data['output']}\n\n"
    return prompt


def gen_emotion_prompt(datas, subject):
    prompt = "Given a comment, please predict the emotion category of this comment. The predict answer must come from the demonstration examples with the exact format.\nThe examples are as follows:\n"
    for data in datas:
        prompt += f"comment: {data['input']}\nemotion category: {data['output']}\n"
    return prompt

def gen_tacred_prompt(datas, subject):
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
    prompt = "Given a sentence and a pair of subject and object entities within the sentence, please predict the relation between the given entities.\nYou can only select from the following words: "
    for i, word in enumerate(candidates):
        if i != len(candidates) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt += "The examples are as follows:\n"
    for data in datas:
        prompt += f"sentence: {data['input']}\nthe relation between the two entities is: {data['output']}\n"
    return prompt

prompt_templates = {"BBH":lambda formula_string, input_string: f"{formula_string}Question: {input_string}\nAnswer:",
"MMLU":lambda formula_string, input_string: f"{formula_string}Question: {input_string}",
"emotion":lambda formula_string, input_string: f"{formula_string}comment: {input_string}\nemotion category:",
"tacred":lambda formula_string, input_string: f"{formula_string}sentence: {input_string}\nthe relation between the two entities is:"
}
prompts = {'BBH':get_BBH_prompt,'MMLU':gen_MMLU_prompt,'emotion':gen_emotion_prompt,'tacred':gen_tacred_prompt}
