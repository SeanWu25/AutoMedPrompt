from data import load_data
from utils import subset, make_loader
from textgrad_wrapper import Prompt_Optimizer
from evaluation_scripts.auto_eval import process_dataset
from textgrad.engine.together import ChatTogether
import os
from dotenv import load_dotenv
import textgrad as tg
import csv
from tqdm import tqdm

load_dotenv()
os.getenv("TOGETHER_API_KEY")


def auto_prompt(model_name, benchmark_name):
    train_set, dev_set, test_set = load_data(benchmark_name)

    dev_set = subset(dev_set)

    train_loader, val_loader, test_loader = make_loader(train_set, dev_set, test_set)
    starting_prompt = "You are a helpful, creative, and smart assistant."
    optimizer_object = Prompt_Optimizer(model_name = model_name, starting_prompt = starting_prompt)
    optimizer_object.train(train_loader,val_loader)
    