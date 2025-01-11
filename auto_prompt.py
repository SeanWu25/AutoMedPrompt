from data import load_data
from utils import subset, make_loader
from textgrad_wrapper import Prompt_Optimizer


def auto_prompt(model_name, benchmark_name, output_dir="C:\\Users\\Admin\\Documents\\autoprompt\\results"):

    train_set, dev_set, test_set = load_data(benchmark_name)
    dev_set = subset(dev_set)

    train_loader, val_loader, test_loader = make_loader(train_set, dev_set, test_set)
    starting_prompt = "Respond to the multiple choice question."
    optimizer_object = Prompt_Optimizer(model_name = model_name, starting_prompt = starting_prompt)

    optimized_prompt = optimizer_object.train(train_loader,val_loader)

    print("*" * 50)
    print("Optimized System Prompt is: ", optimized_prompt)
    print("*" * 50)

    print("*" * 50)
    print("Now evaluating testing set with new prompt: ")

    csv_filename = f"{output_dir}/llama3-70BChat_{benchmark_name}_autoprompt_textgrad_results.csv"

    optimizer_object.eval(test_loader, csv_file_path = csv_filename)







    
 

 