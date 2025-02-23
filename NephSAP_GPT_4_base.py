import os
import csv
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from data import load_data

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

def gpt_4_zero_shot(model_name="gpt-4", output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    benchmark_name = "NephSAP"
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_GPT_4_base_results.csv")

    _, _, test_set = load_data("NephSAP")

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["Question", "Ground Truth", "Prediction", "Status"])

        for question_string in tqdm(test_set, desc="Processing Questions", unit="question"):
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
            ground_truth = question_string['answer_idx']

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI answering multiple-choice questions."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.4
                )
                prediction = response.choices[0].message.content
                status = "Success"
            except Exception as e:
                prediction = f"Error: {str(e)}"
                status = "Failed"

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")


if __name__ == "__main__":
    gpt_4_zero_shot()
