import argparse
import os
import pandas as pd
from evaluation_scripts.auto_eval import process_dataset

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--csv_folder", type=str, help="Path to the folder containing CSV files.")
    args = parser.parse_args()

    csv_folder = args.csv_folder

    if not os.path.isdir(csv_folder):
        print(f"The provided path '{csv_folder}' is not a valid directory.")
        return

    for file_name in os.listdir(csv_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_folder, file_name)
            print(f"Processing file: {file_name}")

            df = pd.read_csv(file_path)

            accuracy, total_correct, total_questions = process_dataset(df)

            results_summary = f"""
            Results Summary for {file_name}:
            ------------------------------
            Accuracy       : {accuracy * 100:.2f}%
            Total Correct  : {total_correct}
            Total Questions: {total_questions}
            """
            print(results_summary)

if __name__ == "__main__":
    main()
