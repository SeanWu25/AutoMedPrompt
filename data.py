import json
import random
import pandas as pd
def split_train_dev(train_dev_examples):
    #randomly select 50 examples for dev set
    dev_examples = random.sample(train_dev_examples, 50)
    train_examples = [example for example in train_dev_examples if example not in dev_examples]
    return train_examples, dev_examples


def load_json(bench):
    def load_file(file_path):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    examples.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
        return examples
    
    def load_csv(file_path): #i want this function to load rows in a csv file folowing the logic of the load_file function
        examples = []
       
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            examples.append(row.to_dict())
        
        return examples


    if bench == "MedQA4":
        base_path = "C:\\Users\\Admin\\Documents\\autoprompt\\benchmarks\\MedQA4\\"
        train_dev_examples = load_file(base_path + "train.jsonl")
        train_examples, dev_examples = split_train_dev(train_dev_examples)
      #  dev_examples = load_file(base_path + "dev.jsonl")
        test_examples = load_file(base_path + "test.jsonl")

        return train_examples, dev_examples, test_examples
    if bench == "PubMedQA":

        base_path = "C:\\Users\\Admin\\Documents\\autoprompt\\benchmarks\\PubMedQA\\"

        train_dev_examples = load_csv(base_path + "train_dev.csv")
        train_examples,dev_examples = split_train_dev(train_dev_examples)
        test_examples = load_csv(base_path + "test.csv")


        return train_examples, dev_examples, test_examples

    
def load_data(benchmark):
    train_set, val_set, test_set = load_json(benchmark)
    return train_set, val_set, test_set