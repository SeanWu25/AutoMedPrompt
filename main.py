import argparse
from baseline_methods import zero_shot, few_shot, CoT
from auto_prompt import auto_prompt
def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--model", type = str)
    args = parser.parse_args()

    benchmark = args.benchmark  # medQA, pubmedQA, etc
    method = args.method        # can be zero-shot, few-shot, CoT, etc, or auto_prompt
    model = args.model          # open sourced LLM choice

    # Map methods to corresponding functions
    methods = {
        "zero-shot": zero_shot,
        "few-shot": few_shot,
        "CoT": CoT,
        "auto-prompt": auto_prompt
    }
    method_function = methods.get(method)
    if method_function:
        method_function(model, benchmark)
    else:
        print(f"Error: Unknown method '{method}'. Please choose from {', '.join(methods.keys())}.")

if __name__ == "__main__":
    main()
