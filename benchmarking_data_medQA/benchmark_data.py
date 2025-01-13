import pandas as pd
import re

def extract_answer(prediction):
    """
    Extract the answer (A-E) from the prediction text.
    """
    match = re.search(r"\b[A-E]\b", prediction)
    return match.group(0) if match else None

def process_dataset(df, name):
    """
    Process a dataset to clean predictions and calculate accuracy.
    """
    df['Cleaned Prediction'] = df['Prediction'].apply(extract_answer)
    df['Correct'] = df['Cleaned Prediction'] == df['Ground Truth']
    accuracy = df['Correct'].mean()
    total_correct = df['Correct'].sum()
    total_questions = len(df)
    return accuracy, total_correct, total_questions, df

def main():
    # File paths
    file_zero_shot = 'input_files/llama3-70BChat_MedQA_zero_shot_results.csv'
    file_few_shot = 'input_files/llama3-70BChat_MedQA_few_shot_results.csv'
    file_chain_of_thought = 'input_files/llama3-70BChat_MedQA_chain_of_thought_results.csv'

    # Load datasets
    zero_shot_df = pd.read_csv(file_zero_shot)
    few_shot_df = pd.read_csv(file_few_shot)
    chain_of_thought_df = pd.read_csv(file_chain_of_thought)

    # Process datasets
    zero_shot_accuracy, zero_shot_correct, zero_shot_total, zero_shot_df = process_dataset(zero_shot_df, "Zero Shot")
    few_shot_accuracy, few_shot_correct, few_shot_total, few_shot_df = process_dataset(few_shot_df, "Few Shot")
    chain_of_thought_accuracy, chain_of_thought_correct, chain_of_thought_total, chain_of_thought_df = process_dataset(chain_of_thought_df, "Chain of Thought")

    # Summary results
    results_summary = pd.DataFrame({
        "Dataset": ["Zero Shot", "Few Shot", "Chain of Thought"],
        "Accuracy": [zero_shot_accuracy, few_shot_accuracy, chain_of_thought_accuracy],
        "Total Correct": [zero_shot_correct, few_shot_correct, chain_of_thought_correct],
        "Total Questions": [zero_shot_total, few_shot_total, chain_of_thought_total]
    })

    # Save detailed outputs
    zero_shot_df.to_csv('output_files/zero_shot_processed.csv', index=False)
    few_shot_df.to_csv('output_files/few_shot_processed.csv', index=False)
    chain_of_thought_df.to_csv('output_files/chain_of_thought_processed.csv', index=False)

    # Save and display summary
    results_summary.to_csv('benchmark_results_summary.csv', index=False)
    print("Benchmarking completed. Results summary:")
    print(results_summary)

if __name__ == "__main__":
    main()
