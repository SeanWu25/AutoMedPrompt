import argparse
import json
from data import load_data
from textgrad.engine.together import ChatTogether
import os
from dotenv import load_dotenv
import textgrad as tg
import csv
from tqdm import tqdm

load_dotenv()
os.getenv("TOGETHER_API_KEY")

def yn_post_eval(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results", system_prompt = "Respond to the multiple choice question."):
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_auto_prompt_text_grad.csv")

    model = ChatTogether(model_name, system_prompt)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            question = (
                f"Context: {question_string['CONTEXTS']}\n"
                f"Question: {question_string['QUESTION']}\n"
            )

            ground_truth = question_string['final_decision']

            prediction = model.generate(question,temperature=0.4)
            status = "Success"

            writer.writerow([question_string['QUESTION'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
def post_eval(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results", system_prompt = "Respond to the multiple choice question."):
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_auto_prompt_text_grad.csv")

    model = ChatTogether(model_name, system_prompt)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            question = (
                f"Question: {question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                f"E. {question_string['options']['E']}\n"
            )

        

            ground_truth = question_string['answer_idx']

            prediction = model.generate(question,temperature=0.4)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
 
def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--json_path", type=str, help="Json file path")
    parser.add_argument("--benchmark_name", type=str, help="Benchmark name")
    args = parser.parse_args()

    file_path = args.json_path
    benchmark_name = args.benchmark_name
    print("*" * 50)
    print("SET TO EVALUATION MODE")
    print("*" * 50)

    with open(file_path, "r") as file:
        data = json.load(file)
    last_event = data["events"][-1]  
    model_name = data["events"][0].get("model_name")
    system_prompt = last_event.get("updated_prompt", "N/A") 
    print("-" * 50)
    print("BENCHMARKING: ")
    print(f"Model Name: {model_name}")
    print(f"System Prompt: {system_prompt}")
    print("-" * 50)


    if benchmark_name == "MedQA":
        post_eval(model_name= model_name, benchmark_name = benchmark_name, system_prompt = system_prompt)
    elif benchmark_name == "PubMedQA":
        yn_post_eval(model_name= model_name, benchmark_name = benchmark_name, system_prompt = system_prompt)













    
 

 

if __name__ == "__main__":
    main()
