import json

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

    if bench == "MedQA":
        base_path = "C:\\Users\\Admin\\Documents\\autoprompt\\benchmarks\\MedQA\\"
        train_examples = load_file(base_path + "train.jsonl")
        dev_examples = load_file(base_path + "dev.jsonl")
        test_examples = load_file(base_path + "test.jsonl")

        return train_examples, dev_examples, test_examples

    
def load_data(benchmark):
    train_set, val_set, test_set = load_json(benchmark)
    return train_set, val_set, test_set