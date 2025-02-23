import os
from dotenv import load_dotenv
from textgrad.engine.groq import ChatGroq
from textgrad.engine.together import ChatTogether
import textgrad as tg
import csv 
import random
from data import load_data
from tqdm import tqdm

load_dotenv()
os.getenv("TOGETHER_API_KEY")

def zero_shot(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_zero_shot_results.csv")

    model = ChatTogether(model_name)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            if benchmark_name == "MedQA4":
                question = (
                f"Question: {question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                )
            elif benchmark_name == "NephSAP":
                
                question = (
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

                print(question)
                print()



            ground_truth = question_string['answer_idx']

            prediction = model.generate(question, temperature=0.4)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
 

def few_shot(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    def process_in_context(example_string):
        if benchmark_name == "MedQA4":
            question_text = (
                f"Question: {example_string['question']}\n"
                f"Options:\n"
                f"A. {example_string['options']['A']}\n"
                f"B. {example_string['options']['B']}\n"
                f"C. {example_string['options']['C']}\n"
                f"D. {example_string['options']['D']}\n"
            )
        elif benchmark_name == "NephSAP":
            question_text = (
                f"Context: {example_string['context']}\n" if 'context' in example_string else ''
            ) + (
                f"Question: {example_string['question']}\n"
                f"Options:\n"
                f"A. {example_string['options']['A']}\n"
                f"B. {example_string['options']['B']}\n"
                f"C. {example_string['options']['C']}\n"
    f"{f'D. {question_string['options']['D']}\n' if 'D' in question_string['options'] else ''}"
                f"{f'E. {example_string['options']['E']}\n' if 'E' in example_string['options'] else ''}"
            )

        correct_answer = example_string['answer']

        
        formatted_output = (
            f"{question_text}\n"
            f"Correct Answer is: {example_string['answer_idx']}, {correct_answer}"
        )
        return formatted_output
    
    def select_random_shot(train_set):
        return random.sample(train_set, 3)


    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_few_shot_results.csv")

    model = ChatTogether(model_name)
    train_set, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])
        


        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):

            icl1,icl2,icl3 = select_random_shot(train_set)

            in_context1, in_context2, in_context3 = process_in_context(icl1), process_in_context(icl2),process_in_context(icl3)

            formatted_context = (
                f"Example 1:\n{in_context1}\n\n"
                f"Example 2:\n{in_context2}\n\n"
                f"Example 3:\n{in_context3}\n\n"
            )
            if benchmark_name == "MedQA4":
                question_text = (
                    f"Question: {question_string['question']}\n"
                    f"Options:\n"
                    f"A. {question_string['options']['A']}\n"
                    f"B. {question_string['options']['B']}\n"
                    f"C. {question_string['options']['C']}\n"
                    f"D. {question_string['options']['D']}\n"
                )
            elif benchmark_name == "NephSAP":
                question_text = (
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

            question = f"{formatted_context}Now, answer the following question:\n\n{question_text}"

            print(question)
            print()
            print()
            print()

            ground_truth = question_string['answer_idx']

            prediction = model.generate(question, temperature=0.4)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
 


def CoT(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_chain_of_thought_results.csv")
    deep_seek_r1 = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    model = ChatTogether(model_name, system_prompt = deep_seek_r1)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Reasoning", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            if benchmark_name == "MedQA4":
                question = (
                f"Question: {question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                )
            elif benchmark_name == "NephSAP":
                question = (
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

            print(question)
            print()
            print()
         
            ground_truth = question_string['answer_idx']

            prediction = model.generate(question, temperature=0.4)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction,status])

    print("BENCHMARKING COMPLETE")
    return

  




