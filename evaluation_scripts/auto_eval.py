import pandas as pd
import re

def extract_answer(prediction):
    """
    Extract the answer (A-E) from the prediction text.
    """
    match = re.search(r"\b[A-E]\b", prediction)

    return match.group(0) if match else None

def process_dataset(df):
    """
    Process a dataset to clean predictions and calculate accuracy.
    """
    df['Cleaned Prediction'] = df['Prediction'].apply(extract_answer)
    df['Correct'] = df['Cleaned Prediction'] == df['Ground Truth']
    accuracy = df['Correct'].mean()
    total_correct = df['Correct'].sum()
    total_questions = len(df)
    return accuracy, total_correct, total_questions

