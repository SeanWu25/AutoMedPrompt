from data import load_data
from utils import subset, make_loader
from textgrad_wrapper import Prompt_Optimizer
from evaluation_scripts.auto_eval import process_dataset
from textgrad.engine.together import ChatTogether
import os
from dotenv import load_dotenv
import textgrad as tg
import csv
from tqdm import tqdm

load_dotenv()
os.getenv("TOGETHER_API_KEY")

def post_eval(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results", system_prompt = "Respond to the multiple choice question."):
    csv_filename = f"{output_dir}/llama3-70BChat_{benchmark_name}_autoprompt_textgrad_results.csv"

    engine = ChatTogether(model_name)
    model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            question_text = (
                f"Question: {question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                f"E. {question_string['options']['E']}\n"
            )

            question = tg.Variable(
                question_text,
                role_description="question to the LLM",
                requires_grad=False
            )

            ground_truth = question_string['answer_idx']

            prediction = model(question)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
 
def auto_prompt(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results", post = False):
    train_set, dev_set, test_set = load_data(benchmark_name)

    if post == False:
        print("reached here?")
        dev_set = subset(dev_set)

        train_loader, val_loader, test_loader = make_loader(train_set, dev_set, test_set)
        starting_prompt = "Respond to the multiple choice question."
        optimizer_object = Prompt_Optimizer(model_name = model_name, starting_prompt = starting_prompt)
        optimizer_object.train(train_loader,val_loader)
    elif post == True:
        import json
        print("*" * 50)
        print("SET TO EVALUATION MODE")
        print("*" * 50)

        file_path = "C:\\Users\\Admin\\Documents\\autoprompt\\prompt_logs\\training_log_20250113_171127.json"
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
        post_eval(model_name= model_name, benchmark_name = benchmark_name, system_prompt = system_prompt)














    
 

 