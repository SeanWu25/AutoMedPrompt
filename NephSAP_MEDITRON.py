import os
import csv
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from data import load_data
import replicate

load_dotenv()
api_key = os.getenv("REPLICATE_API_TOKEN")

os.environ["REPLICATE_API_TOKEN"] =api_key


def meditron(model_name="Meditron",benchmark_name = "NephSAP", output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):
    benchmark_name = "NephSAP"
    model_name_part = model_name.split("/")[1] if "/" in model_name else model_name
    model_output_dir = os.path.join(output_dir, model_name_part, benchmark_name)
    os.makedirs(model_output_dir, exist_ok=True)  
    csv_filename = os.path.join(model_output_dir, f"{benchmark_name}_MEDITRON_results.csv")

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

            print("RUNNING LLM NOW ")
            response = replicate.run(
                "titocosta/meditron-70b-awq:7cbcd02ebd1baa7f800969f60dada8bd33c30e8d467223ce78842ecba2fbbc86",
                input={
                    "temperature": 0.4,
                    "system_message": "You are a helpful AI assistant trained in the medical domain",
                    "prompt_template": "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>question\n{prompt}<|im_end|>\n<|im_start|>answer\n",
                    "prompt": question  # Include the constructed question prompt
                }
            )
            print("FINISHED RUNNING LLM")
            response_list = list(response)
            

            print(response_list)
            print(type(response_list))

            # Extract prediction safely
            prediction = response_list[0] if response_list else None
            raise Exception("Error: This is a test error")
            print("Prediction: ", prediction)
            print("*" * 50)
            status = "Success"
        

            writer.writerow([question_string['question'], ground_truth, prediction, status])

    print("BENCHMARKING COMPLETE")


if __name__ == "__main__":
    meditron(model_name="Meditron", benchmark_name = "NephSAP")