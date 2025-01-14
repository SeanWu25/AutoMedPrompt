import argparse
from evaluation_scripts.auto_eval import process_dataset
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--csv_path", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    csv_path = args.csv_path

    df = pd.read_csv(csv_path)

    accuracy, total_correct, total_questions = process_dataset(df)

    results_summary = f"""
    Results Summary:
    ----------------
    Accuracy       : {accuracy * 100:.2f}%
    Total Correct  : {total_correct}
    Total Questions: {total_questions}
    """
    print(results_summary)

if __name__ == "__main__":
    main()
