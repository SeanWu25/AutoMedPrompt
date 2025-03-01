import pandas as pd
import re


def extract_answer(prediction):
    """
    Extract the answer (A-E) from the prediction text, including from <answer> tags.
    """
    prediction = str(prediction)
    tag_match = re.search(r"<answer>\s*([A-E])\b", prediction)
    if tag_match:
        return tag_match.group(1)
    

    #match = re.search(r'(?:Answer\s+)?([A-E])(?:[\s:.]|[\)\s]|\b)', prediction)

    match = re.search(r"\b[A-E]\b", prediction)

    return match.group(0) if match else None

def yn_extract_answer(prediction):
    # Attempt to extract from <answer> tags
    tag_match = re.search(r"<answer>\s*([A-Za-z]+)\s*</answer>", prediction, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).strip().lower()

    # Attempt to extract from phrases like "answer is yes"
    phrase_match = re.search(r"answer is\s*([A-Za-z]+)", prediction, re.IGNORECASE)
    if phrase_match:
        return phrase_match.group(1).strip().lower()

    # Fallback: Plain text answers like "yes", "no", "maybe"
    match = re.search(r"\b(yes|no|maybe)\b", prediction, re.IGNORECASE)
    return match.group(1).strip().lower() if match else None

def process_dataset(df, benchmark_name):
    """
    Process a dataset to clean predictions and calculate accuracy.
    """
    if benchmark_name=="MedQA4" or benchmark_name=="NephSAP":
        df['Cleaned Prediction'] = df['Prediction'].apply(extract_answer)
    elif benchmark_name == "PubMedQA":
        df['Cleaned Prediction'] = df['Prediction'].apply(yn_extract_answer)
        #how to print the predicted and the ground truth together side by side?
        print(df[['Prediction', 'Ground Truth']])

    df['Correct'] = df['Cleaned Prediction'] == df['Ground Truth']
    accuracy = df['Correct'].mean()
    total_correct = df['Correct'].sum()
    total_questions = len(df)
    return accuracy, total_correct, total_questions

