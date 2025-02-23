import json
import re
from openai import OpenAI
from tqdm import tqdm
import os
import re
from dotenv import load_dotenv
from data import load_data
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key= api_key)
def read_jsonl_file(file_path):
    """
    Parses a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list of dict: A list where each element is a dictionary representing
            a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

def write_jsonl_file(file_path, dict_list):
    with open(file_path, 'w') as file:
        for dictionary in dict_list:
            # Convert the dictionary to a JSON string and write it to the file.
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')


def create_query(question_string):

    query = (
        f"Context: {question_string['context']}\n" if 'context' in question_string else ''
    ) + (
        f"Question: {question_string['question']}\n"
        f"Options:\n"
        f"A. {question_string['options']['A']}\n"
        f"B. {question_string['options']['B']}\n"
        f"C. {question_string['options']['C']}\n"
        f"{f'D. {question_string['options']['D']}\n' if 'D' in question_string['options'] else ''}"
        f"{f'E. {question_string['options']['E']}\n' if 'E' in question_string['options'] else ''}"
    )
    return query

def build_zero_shot_prompt(system_prompt, question):

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": create_query(question)}]
    return messages

def format_answer(cot, answer):
    return f"""## Answer
{cot}
Therefore, the answer is {answer}"""

def build_few_shot_prompt(system_prompt, question, examples, include_cot=True):

    messages = [{"role": "system", "content": system_prompt}]
    
    for elem in examples:
        messages.append({"role": "user", "content": create_query(elem)})
        if include_cot:
            messages.append({"role": "assistant", "content": format_answer(elem["cot"], elem["answer_idx"])})        
        else:           
            answer_string = f"""## Answer\nTherefore, the answer is {elem["answer_idx"]}"""
            messages.append({"role": "assistant", "content": answer_string})
            
    messages.append({"role": "user", "content": create_query(question)})
    return messages

def get_response(messages, model_name, temperature = 0.0, max_tokens = 10):

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def matches_ans_option(s):

    return bool(re.match(r'^Therefore, the answer is [A-Z]', s))

def extract_ans_option(s):

    match = re.search(r'^Therefore, the answer is ([A-Z])', s)
    if match:
        return match.group(1)  # Returns the captured alphabet
    return None 

def matches_answer_start(s):

    return s.startswith("## Answer")

def validate_response(s):

    file_content = s.split("\n")
    
    return matches_ans_option(file_content[-1]) and matches_answer_start(s)

def parse_answer(response):
 
    split_response = response.split("\n")
    assert split_response[0] == "## Answer"
    cot_reasoning = "\n".join(split_response[1:-1]).strip()
    ans_choice = extract_ans_option(split_response[-1])
    return cot_reasoning, ans_choice


system_prompt = """You are an expert medical professional. You are provided with a medical question with multiple answer choices.
Your goal is to think through the question carefully and explain your reasoning step by step before selecting the final answer.
Respond only with the reasoning steps and answer as specified below.
Below is the format for each question and answer:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
(model generated chain of thought explanation)
Therefore, the answer is [final model answer (e.g. A,B,C,D,E)]"""

system_zero_shot_prompt = """You are an expert medical professional. You are provided with a medical question with multiple answer choices.
Your goal is to think through the question carefully and respond directly with the answer option.
Below is the format for each question and answer:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
Therefore, the answer is [final model answer (e.g. A,B,C,D,E)]"""

def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

'''

train_data = read_jsonl_file("C:\\Users\\Admin\\Downloads\\nephsap_train.jsonl")
cot_responses = []
os.mkdir("cot_responses")

for idx, item in enumerate(tqdm(train_data)):    
    prompt = build_zero_shot_prompt(system_prompt, item)
    try:
        response = get_response(prompt, model_name="gpt-4", max_tokens=500)
        cot_responses.append(response)
        with open(os.path.join("cot_responses", str(idx) + ".txt"), "w", encoding="utf-8") as f:
            f.write(response)           
    except Exception as e :
        print(str(e))
        cot_responses.append("")

questions_dict = []
ctr = 0
for idx, question in enumerate(tqdm(train_data)):
    file =  open(os.path.join("cot_responses", str(idx) + ".txt"), encoding="utf-8").read()
    if not validate_response(file):
        continue
    
    cot, pred_ans = parse_answer(file)
    
    dict_elem = {}
    dict_elem["idx"] = idx
    dict_elem["question"] = question["question"]
    dict_elem["answer"] = question["answer"]
    dict_elem["options"] = question["options"]
    dict_elem["cot"] = cot
    dict_elem["pred_ans"] = pred_ans
    questions_dict.append(dict_elem)        


write_jsonl_file("C:\\Users\\Admin\\Downloads\\cot_responses_medqa_train_set.jsonl", questions_dict)
'''
'''
questions_dict = read_jsonl_file("C:\\Users\\Admin\\Downloads\\cleaned_medqa_train_set_direct.jsonl")

filtered_questions_dict = []
for item in tqdm(questions_dict):
    print()
    pred_ans = item["pred_ans"]  # Get the actual answer text from the predicted label
    if pred_ans == item["answer_idx"]:
        filtered_questions_dict.append(item)

write_jsonl_file("C:\\Users\\Admin\\Downloads\\cot_responses_medqa_train_set_filtered.jsonl", filtered_questions_dict)


for item in tqdm(filtered_questions_dict):
    #the item["context"] and item["question"] should be together in the get_embedding
    combined = item["context"] + "\n" + item["question"]
    item["embedding"] = get_embedding(item["question"])
    inv_options_map = {v:k for k,v in item["options"].items()}
    item["answer_idx"] = inv_options_map[item["answer"]]    

write_jsonl_file("C:\\Users\\Admin\\Downloads\\cot_responses_medqa_train_set_filtered_with_embeddings.jsonl", filtered_questions_dict)

'''

filtered_questions_dict = read_jsonl_file("C:\\Users\\Admin\\Downloads\\cot_responses_medqa_train_set_filtered_with_embeddings.jsonl")

embeddings = np.array([d["embedding"] for d in filtered_questions_dict])
indices = list(range(len(filtered_questions_dict)))

knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine').fit(embeddings)
print("*" * 50)
print("NOW EVALUATING ON TEST SET")
print("*" * 50)
test_samples = read_jsonl_file("C:\\Users\\Admin\\Downloads\\direct_medqa_test_set.jsonl")

def shuffle_option_labels(answer_options):
    options = list(answer_options.values())
    random.shuffle(options)
    labels = [chr(i) for i in range(ord('A'), ord('A') + len(options))]
    return {label: option for label, option in zip(labels, options)}

for question in tqdm(test_samples, colour="green"):
    question_variants = []
    prompt_variants = []
    cot_responses = []

    question_embedding = get_embedding(question["context"] + " \n" + question["question"])
    distances, top_k_indices = knn.kneighbors([question_embedding], n_neighbors=5)
    top_k_dicts = [filtered_questions_dict[i] for i in top_k_indices[0]]
    question["outputs"] = []

    for idx in range(5):
        question_copy = question.copy()
        shuffled_options = shuffle_option_labels(question["options"])
        inv_map = {v:k for k,v in shuffled_options.items()}

        question_copy["options"] = shuffled_options
        question_copy["answer_idx"] = inv_map[question_copy["answer"]]
        question_variants.append(question_copy)
        prompt = build_few_shot_prompt(system_prompt, question_copy, top_k_dicts)
        prompt_variants.append(prompt)

    for prompt in tqdm(prompt_variants):
        response = get_response(prompt, model_name="gpt-4", max_tokens=500)
        cot_responses.append(response)

    for question_sample, answer in zip(question_variants, cot_responses):
        if validate_response(answer):
            cot, pred_ans = parse_answer(answer)
        else:
            cot = ""
            pred_ans = ""

        question["outputs"].append({
            "question": question_sample["question"],
            "options": question_sample["options"],
            "cot": cot,
            "pred_ans": question_sample["options"].get(pred_ans, "")
        })

write_jsonl_file("C:\\Users\\Admin\\Downloads\\final_processed_test_set_responses_medprompt.jsonl", test_samples)


test_samples = read_jsonl_file("C:\\Users\\Admin\\Downloads\\final_processed_test_set_responses_medprompt.jsonl")
from collections import Counter

def find_mode_string_list(string_list):
    if not string_list:
        return None  
    string_counts = Counter(string_list)
    max_freq = max(string_counts.values())
    return [string for string, count in string_counts.items() if count == max_freq]

correct_count = 0
for idx, item in enumerate(test_samples):
    print(idx)
    pred_ans = [x["pred_ans"] for x in item["outputs"]]
    freq_ans = find_mode_string_list(pred_ans)

    if len(freq_ans) > 1:
        final_prediction = ""
    else:
        final_prediction = freq_ans[0]

    if final_prediction == item["answer"]:
        correct_count += 1

accuracy = correct_count / len(test_samples)
print(f"Accuracy: {accuracy:.2%}")


