import pandas as pd
import re


def extract_answer(prediction):
    """
    Extract the answer (A-E) from the prediction text, including from <answer> tags.
    """
    tag_match = re.search(r"<answer>\s*([A-E])\b", prediction)
    if tag_match:
        return tag_match.group(1)
    
    match = re.search(r"\b[A-E]\b", prediction)
    return match.group(0) if match else None

def yn_extract_answer(prediction):
    """
    Extract the answer from the prediction text. Supports answers enclosed in <answer> tags
    and plain text answers like "yes" or "no".
    """
    # Attempt to extract from <answer> tags
    match = re.search(r"<answer>\s*(\w+)\s*</answer>", prediction, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()

    # If no tags are found, use the plain text directly
    return prediction.strip().lower()

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

