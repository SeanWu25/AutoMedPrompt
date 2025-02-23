from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.evaluation.schema import StringEvaluator
from textgrad.tasks.base import DataLoader
from typing import Optional, Any
import random
import re
from typing import Optional
import pandas as pd

def subset(dataset, seed = 42):
   # random.seed(seed) 
    return random.sample(dataset, 50)


def yn_loader(train_set, dev_set,test_set, batch_size = 1):
    def json_to_list(set):
        questions = [f"Context: {question['CONTEXTS']}\n Question: {question['QUESTION']}\n" for question in set]
        references = [reference['final_decision'] for reference in set]
        explanation = [reference['LONG_ANSWER'] for reference in set]
        return list(zip(questions,references,explanation))

    print("*" * 50)
    train_set = json_to_list(train_set)
    dev_set = json_to_list(dev_set)
    test_set = json_to_list(test_set)

    print("Length of train set: ", len(train_set))
    print("Length of dev set ", len(dev_set))
    print("Length of test set ", len(test_set))

    print("*"* 50)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def neph_loader(train_set, dev_set,test_set, batch_size = 5):
    def json_to_list(set):
        questions = [
            (
                (f"Context: {question['context']}\n" if 'context' in question else '') +
                f"Question: {question['question']}\n"
                "Options:\n"
                f"A. {question['options']['A']}\n"
                f"B.  {question['options']['B']}\n"
                f"C. {question['options']['C']}\n"
                + (f"D. {question['options']['D']}\n" if 'D' in question['options'] else '')
                + (f"E. {question['options']['E']}\n" if 'E' in question['options'] else '')
            )
            for question in set
        ]
        references = [reference['answer_idx'] for reference in set]
        explanation = [reference['answer'] for reference in set]
        return list(zip(questions,references,explanation))

    print("*" * 50)
    train_set = json_to_list(train_set)
    dev_set = json_to_list(dev_set)
    test_set = json_to_list(test_set)

    print("Length of train set: ", len(train_set))
    print("Length of dev set ", len(dev_set))
    print("Length of test set ", len(test_set))

    print("*"* 50)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def make_loader(train_set, dev_set,test_set, batch_size = 5):
    def json_to_list(set):
        
        questions = [ f"Question: {question['question']}\n" f"Options:\n" f"A. {question['options']['A']}\n" f"B. {question['options']['B']}\n" f"C. {question['options']['C']}\n" f"D. {question['options']['D']}\n" + (f"E. {question['options']['E']}\n" if 'E' in question['options'] else "") for question in set ]
        references = [reference['answer_idx'] for reference in set]
        explanation = [reference['answer'] for reference in set]
        return list(zip(questions,references,explanation))


    print("*" * 50)
    train_set = json_to_list(train_set)
    dev_set = json_to_list(dev_set)
    test_set = json_to_list(test_set)

    print("Length of train set: ", len(train_set))
    print("Length of dev set ", len(dev_set))
    print("Length of test set ", len(test_set))

    print("*"* 50)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


class MetricsAggregator:
    def __init__(self):
        self.correctness_values = []

    def aggregate(self, metrics):
        """
        This function takes in a dictionary containing the evaluation metrics and
        extracts the 'correctness' value to store it.
        
        Args:
        - metrics (dict): A dictionary with various evaluation metrics, including 'correctness'.
        """
        if "correctness" in metrics:
            self.correctness_values.append(metrics["correctness"])

    def get_aggregated_metrics(self):
        """
        This function returns the average correctness value based on the stored values.

        Returns:
        - float: The average correctness, or None if no values have been aggregated.
        """
        if not self.correctness_values:
            return None
        return sum(self.correctness_values) / len(self.correctness_values)
  

class CustomQAEvaluator(StringEvaluator):

    def __init__(self) -> None:
        """Initialize the evaluator with GPT-4o-mini."""
        self.chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["prediction", "reference"],
            template = """
            Evaluate if the student's answer matches the true answer. Respond with one word only: Correct or Incorrect.
            Student ANSWER: {prediction}
            TRUE ANSWER:{reference}
            GRADE:"""
        )

    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction against the reference using a comprehensive QA approach."""
        prompt = self.prompt_template.format(prediction=prediction, reference=reference)
        
        print("Prompt to evaluate: ", prompt)
        result = self.chat_model([HumanMessage(content=prompt)])

        print("evaluation result: ", result.content.lower())

        if "incorrect" in result.content.lower():
            print("Incorrect")
            return {"score": 0}
        else:
            print("Correct")
            return {"score": 1}
        
   

class RegExCustomQAEvaluator:
    def __init__(self) -> None:
        """
        Initialize the evaluator object. This evaluator uses a regular expression method
        to check if the prediction matches the ground truth.
        """
        pass

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """
        Extract the answer (A-E) from a given text using regular expressions.
        """
        import re
        match = re.search(r"\b[A-E]\b", text)
        return match.group(0) if match else None

    def evaluate(self, prediction: str, ground_truth: str) -> int:
        """
        Evaluate the prediction against the ground truth.
        Returns 1 if the prediction matches the ground truth, otherwise 0.
        """
        pred_answer = self.extract_answer(prediction)
        true_answer = self.extract_answer(ground_truth)



        return 1 if pred_answer == true_answer else 0  
    
class PredictionEvaluator:
    def __init__(self) -> None:
        """
        Initialize the evaluator object. This evaluator checks if the prediction matches
        the ground truth and can extract answers from tagged text.
        """
        pass

    def extract_answer(self,prediction: str) -> Optional[str]:
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

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize answers by converting to lowercase and removing extra spaces.
        """
        return answer.strip().lower()

    def evaluate(self, prediction: str, reference: str) -> int:
        """
        Evaluate a single prediction against the ground truth.
        Extracts the answer from the prediction and compares it to the ground truth.
        Returns 1 if the prediction matches the ground truth, otherwise 0.
        """
        pred = self.extract_answer(prediction) or self.normalize_answer(prediction)
        truth = self.normalize_answer(reference)

        print("prediction: ", pred, "truth: ", truth)
        return 1 if pred == truth else 0