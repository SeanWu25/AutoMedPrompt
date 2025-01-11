import textgrad as tg
from tqdm import tqdm
import os
import csv
from dotenv import load_dotenv
from textgrad.engine.together import ChatTogether
from utils import MetricsAggregator, CustomQAEvaluator
load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("TOGETHER_API_KEY")



backward_engine = tg.get_engine(engine_name="gpt-4o-mini")
tg.set_backward_engine(backward_engine, override=True)


class Prompt_Optimizer:
   def __init__(self, model_name,starting_prompt,patience=5):
       self.patience = patience
       self.no_improvement_steps = 0
       self.model_name = model_name

       self.system_prompt_var = tg.Variable(
           starting_prompt,
           requires_grad=True,
           role_description="A general system prompt for a language model designed to answer medical-related multiple-choice questions."
       )
       optimizer_engine = tg.get_engine(engine_name="gpt-4o-mini")

       self.optimizer = tg.TextualGradientDescent(
           engine=optimizer_engine,
           parameters=[self.system_prompt_var],
       )

       self.previous_prompt_var = tg.Variable(
           starting_prompt,
           requires_grad=False,
           role_description="Previous best system prompt."
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

       if current_score < self.previous_performance:
           self.system_prompt_var.set_value(self.previous_prompt_var.value)
           print("THIS SYSTEM PROMT FAILED: ", self.system_prompt_var.value)
           print("REVERTED SYSTEM PROMPT AND DEMONSTRATIONS TO PREVIOUS BEST.")
           current_score = self.previous_performance
           self.no_improvement_steps += 1
       else:
           self.previous_prompt_var = copy.deepcopy(self.system_prompt_var)
           self.previous_performance = current_score
           self.no_improvement_steps = 0

       print(f"Validation Metrics: {val_metrics}")
       print(f"Updated system prompt: {self.system_prompt_var.value}")
      

   def train_step(self, batch_x, batch_y):
       """
       Perform a training step with the given batch of data.
       """
       losses = []
       self.optimizer.zero_grad()
       for (x, reference) in zip(batch_x, batch_y):
           x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
           response = self._forward(x.value)

           eval_output_variable = self.eval_item(x.value, response, reference)
           losses.append(eval_output_variable)
            
       total_loss = tg.sum(losses)
       total_loss.backward()

       print("System prompt gradients", self.system_prompt_var.gradients)

       self.optimizer.step()
       self._modify_or_set_agent()


   def evaluate_dataset(self, data_loader):
       """
       Perform an evaluation step with the given batch of data and aggregate the metrics.
       """
       aggregator = MetricsAggregator()

       for _, (batch) in enumerate(tqdm(data_loader)):
           batch_x = batch[0]
           batch_y = batch[1]

           for x, reference in zip(batch_x, batch_y):
               response = self._forward(x)
               
               metrics = self.evaluate(x, response, reference)
  
               aggregator.aggregate(metrics)


       overall_metrics = aggregator.get_aggregated_metrics()
       return overall_metrics


   def train(self, train_loader, val_loader):
       step = 0
       for batch in train_loader:
           print(f"Training Step: {step + 1}")
           batch_x = batch[0]
           batch_y = batch[1]
           self.train_step(batch_x, batch_y)
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

   def eval_item(self, question, response, reference) -> tg.Variable:
       evaluation_instruction = (f"Here's a question: {question}. "
                                f"Evaluate any given answer to this question based on the ground truth:{reference}"
                                "be smart, logical, and very critical. "
                                "Just provide concise feedback.")

       loss_fn = tg.TextLoss(evaluation_instruction)
       loss = loss_fn(response)
       return loss
   
   def evaluate(self, question, response, reference):

       qa_evaluator = CustomQAEvaluator()
          
       evaluation_result = qa_evaluator._evaluate_strings(
           prediction=response.value,
           reference=reference,
           input=question
       )
       metrics_dict = {
           "question": question,
           "agent_answer": response.value,
           "correct_answer": reference
       }
       metrics_dict["correctness"] = evaluation_result['score']
       return metrics_dict



