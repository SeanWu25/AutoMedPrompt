import textgrad as tg
from tqdm import tqdm
import os
import csv
from dotenv import load_dotenv
from datetime import datetime
import json
from textgrad.engine.together import ChatTogether
from utils import MetricsAggregator, CustomQAEvaluator, RegExCustomQAEvaluator
load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("TOGETHER_API_KEY")


BACKWARD_ENGINE_NAME = "gpt-4o-mini"
backward_engine = tg.get_engine(engine_name=BACKWARD_ENGINE_NAME)
tg.set_backward_engine(backward_engine, override=True)


class Prompt_Optimizer:
   def __init__(self, model_name,starting_prompt,patience=3, eval_type = "code"):
       self.patience = patience
       self.no_improvement_steps = 0
       self.model_name = model_name

       self.eval_type = eval_type

       self.starting_prompt = starting_prompt
       self.previous_performance = -float('inf')


       #for logging experiments
       self.log_dir = "prompt_logs"
       os.makedirs(self.log_dir, exist_ok=True)


       self.system_prompt_var = tg.Variable(
           starting_prompt,
           requires_grad=True,
           role_description="A general system prompt for a language model designed to answer medical-related multiple-choice questions. This system prompt should not be too verbose."
       )
       self.log_file = self.log_training_start()

       optimizer_engine = tg.get_engine(engine_name=BACKWARD_ENGINE_NAME)

       self.optimizer = tg.TextualGradientDescent(
           engine=optimizer_engine,
           parameters=[self.system_prompt_var],
       )

       self.previous_prompt_var = tg.Variable(
           starting_prompt,
           requires_grad=True,
           role_description="A general system prompt for a language model designed to answer medical-related multiple-choice questions. This system prompt should not be too verbose."
       )
    
       self._modify_or_set_forward_pass()

   def _modify_or_set_forward_pass(self):
       engine = ChatTogether(self.model_name)
       self.model = tg.BlackboxLLM(engine, system_prompt=self.system_prompt_var)
       
   def _forward(self, query):
       question = tg.Variable(
            query,
            role_description="question to the LLM",
            requires_grad=False
        )
       answer = self.model(question)
       answer.set_role_description("LLM response to the multiple choice question.")

       return answer

    
   def _run_validation_revert(self, val_loader): #guiding optimization trajectory
       import copy
       val_metrics = self.evaluate_dataset(val_loader)
       current_score = val_metrics

       if current_score <= self.previous_performance:
           self.system_prompt_var.set_value(self.previous_prompt_var.value)
           print("THIS SYSTEM PROMT FAILED: ", self.system_prompt_var.value)
           print("REVERTED SYSTEM PROMPT AND DEMONSTRATIONS TO PREVIOUS BEST.")
           current_score = self.previous_performance
           self.no_improvement_steps += 1
       else:
           self.previous_prompt_var = copy.deepcopy(self.system_prompt_var)
           self.previous_performance = current_score
           self.no_improvement_steps = 0
           self.log_prompt_update(self.system_prompt_var, val_metrics)  # Log prompt update here


       print(f"Validation Metrics: {val_metrics}")
       print(f"Updated system prompt: {self.system_prompt_var.value}")
      

   def train_step(self, batch_x, batch_y, batch_z):
       """
       Perform a training step with the given batch of data.
       """
       losses = []
       self.optimizer.zero_grad()
       for (x,ground_truth, explanation) in zip(batch_x, batch_y, batch_z):
           x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
           response = self._forward(x.value)


           eval_output_variable = self.eval_item(x, response,ground_truth, explanation)
           losses.append(eval_output_variable)
            
       total_loss = tg.sum(losses)
       total_loss.backward()

       #print("System prompt gradients", self.system_prompt_var.gradients)

       self.optimizer.step()
       self._modify_or_set_forward_pass()


   def evaluate_dataset(self, data_loader):
       """
       Perform an evaluation step with the given batch of data and aggregate the metrics.
       """
       aggregator = MetricsAggregator()

       for _, (batch) in enumerate(tqdm(data_loader)):
           batch_x = batch[0]
           batch_y = batch[1]
           batch_z = batch[2]

           for x, ground_truth, _ in zip(batch_x, batch_y, batch_z):
               response = self._forward(x)
               
               metrics = self.evaluate(x, response, ground_truth)
  
               aggregator.aggregate(metrics)

       overall_metrics = aggregator.get_aggregated_metrics()

       print("Overall Metrics: ")
       print(overall_metrics)
       return overall_metrics


   def train(self, train_loader, val_loader):
       step = 0

       for batch in train_loader:
           print(f"Training Step: {step + 1}")
           batch_x = batch[0]
           batch_y = batch[1]
           batch_z = batch[2]

           self.train_step(batch_x, batch_y, batch_z)
           self._run_validation_revert(val_loader)

           if self.no_improvement_steps >= self.patience:
               print(f"Stopping early at step {step + 1} due to no improvement for 3 consecutive steps.")
               break

           step += 1

       return self.system_prompt_var.value
   
   def eval(self, test_loader, csv_file_path):
       file_exists = os.path.isfile(csv_file_path)
       with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
          writer = csv.writer(csv_file)
          if not file_exists:
             writer.writerow(["Question", "Ground Truth", "Prediction", "Updated Prompt"])

          for batch in test_loader:
             batch_x = batch[0]
             batch_y = batch[1]

             x = tg.Variable(batch_x, requires_grad=False, role_description="query to the language model")
             prediction = self._forward(x.value)

             writer.writerow([batch_x, batch_y, prediction, self.system_prompt_var.value])
    
       print("FINISHED TESTING ON TEST SET!")


   def eval_item(self, response: tg.Variable, ground_truth: str, reference: str, question: tg.Variable) -> tg.Variable:
       evaluation_instruction = tg.Variable(
            "Please evaluate the response provided by the LLM for the medical multiple choice question based on the ground truth answer. "
            "Be smart, logical, and very critical. "
            "Just provide concise feedback.",
            role_description="evaluation instruction"
        )

       role_descriptions = [
            "Language Model Response",
            "Ground Truth Answer Choice",
            "Correct Explanation"
        ]

       loss_fn = tg.loss.MultiFieldEvaluation(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=role_descriptions
        )

       #ground_truth_variable = tg.Variable(ground_truth, requires_grad=False, role_description="Ground Truth Answer Choice")
       reference_variable = tg.Variable(reference, requires_grad=False, role_description="Correct Explanation")
       response.set_role_description("Language Model Response")

       inputs = [response, ground_truth, reference_variable]

       loss = loss_fn(inputs)

       print("*" * 50)
       print(loss)
       return loss


   
   def evaluate(self, question, response, reference):
       metrics_dict = {
           "question": question,
           "agent_answer": response.value,
           "correct_answer": reference
       }
       if self.eval_type == "code":
            qa_evaluator = RegExCustomQAEvaluator()
            evaluation_result = qa_evaluator.evaluate(
                prediction=response.value,
                ground_truth=reference,
            )
            metrics_dict["correctness"] = evaluation_result

       else:
            qa_evaluator = CustomQAEvaluator()
                
            evaluation_result = qa_evaluator._evaluate_strings(
                prediction=response.value,
                reference=reference,
            )
            metrics_dict["correctness"] = evaluation_result['score']

       return metrics_dict
   
   def log_training_start(self):
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": "Training Start",
            "model_name": self.model_name,
            "starting_prompt": self.starting_prompt,
            "prompt_role_desc: ":  self.system_prompt_var.role_description,
            "Backward_engine": BACKWARD_ENGINE_NAME
        }
        log_filename = os.path.join(
            self.log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(log_filename, "w") as log_file:
            json.dump({"events": [log_data]}, log_file, indent=4)
        return log_filename

   def log_prompt_update(self, updated_prompt, val_accuracy):
        

        gradients_serializable = (
        [str(grad.value) for grad in updated_prompt.gradients]
        if updated_prompt.gradients is not None
        else None
        )
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": "System Prompt Updated",
            "updated_prompt": str(updated_prompt.value),
            "system_prompt_gradients": list(gradients_serializable),
            "validation_accuracy (soft)": str(val_accuracy)
        }
        if self.log_file:
            with open(self.log_file, "r+") as log_file:
                data = json.load(log_file)
                data["events"].append(log_data)
                log_file.seek(0)
                json.dump(data, log_file, indent=4)




