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
os.getenv("GROQ_API_KEY")
os.getenv("TOGETHER_API_KEY")

def zero_shot(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    csv_filename = f"{output_dir}/llama3-70BChat_{benchmark_name}_zero_shot_results.csv"

    system_prompt = "Respond to the multiple choice question."
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
 

def few_shot(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    def process_in_context(example_string):
        question_text = (
                f"Question: {example_string['question']}\n"
                f"Options:\n"
                f"A. {example_string['options']['A']}\n"
                f"B. {example_string['options']['B']}\n"
                f"C. {example_string['options']['C']}\n"
                f"D. {example_string['options']['D']}\n"
                f"E. {example_string['options']['E']}\n"
            )

        correct_answer = example_string['answer']

        formatted_output = (
            f"{question_text}\n"
            f"Correct Answer is: {example_string['answer_idx']}, {correct_answer}"
        )
        return formatted_output
    
    def select_random_shot(train_set):
        return random.sample(train_set, 3)


    csv_filename = f"{output_dir}/llama3-70BChat_{benchmark_name}_few_shot_results.csv"
    system_prompt = "Respond to the multiple choice question. A few examples are provided."

    engine = ChatTogether(model_name)
    model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
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
            question_text = (
                f"Question: {question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                f"E. {question_string['options']['E']}\n"
            )

            final_prompt = f"{formatted_context}Now, answer the following question:\n\n{question_text}"

            question = tg.Variable(
                final_prompt,
                role_description="question to the LLM",
                requires_grad=False
            )

            ground_truth = question_string['answer_idx']

            prediction = model(question)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")
    return
 


def CoT(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    csv_filename = f"{output_dir}/llama3-70BChat_{benchmark_name}_chain_of_thought_results.csv"

    system_prompt = (
        "You are a skilled physician answering medical questions. For each question, carefully analyze the symptoms, "
        "clinical presentation, and provided options. Explain your reasoning step-by-step as you would when diagnosing "
        "a patient, and then provide the most appropriate answer."
    )
    engine = ChatTogether(model_name)
    model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
    _, _, test_set = load_data(benchmark_name)

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Reasoning", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
            question_text = (
                f"Patient Case:\n{question_string['question']}\n"
                f"Options:\n"
                f"A. {question_string['options']['A']}\n"
                f"B. {question_string['options']['B']}\n"
                f"C. {question_string['options']['C']}\n"
                f"D. {question_string['options']['D']}\n"
                f"E. {question_string['options']['E']}\n"
            )

            final_prompt = (
                f"{question_text}\n"
                f"Simulate the diagnostic reasoning of a physician. Consider the patient's symptoms and history step-by-step, "
                f"eliminate incorrect options, and select the most plausible diagnosis or answer."
            )



            question = tg.Variable(
                final_prompt,
                role_description="medical diagnostic question with step-by-step reasoning",
                requires_grad=False
            )

            ground_truth = question_string['answer_idx']

            prediction = model(question)
            status = "Success"

            writer.writerow([question_string['question'], ground_truth, prediction,status])

    print("BENCHMARKING COMPLETE")
    return

  




