from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.evaluation.schema import StringEvaluator
from textgrad.tasks.base import DataLoader
from typing import Optional, Any
import random

def subset(dataset, seed = 42):
    random.seed(seed) 
    return random.sample(dataset, 5)

def make_loader(train_set, dev_set,test_set, batch_size = 1):
    def json_to_list(set):
        questions = [question['question'] for question in set]
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
        
   