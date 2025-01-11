# Automating Prompt Engineering for Medicine

Prompt engineering is key to enhancing the performance of large language models (LLMs). Traditional methods, while effective, often rely on manual effort or require adding additional information to the context window. This project introduces a novel automated framework, using **TextGrad**, to optimize LLMs for medical benchmarks with minimal intervention.

## About the Project

This framework streamlines prompt engineering by automating the process. Traditional in-context learning, which typically involves providing a few examples, may not effectively leverage broader dataset trends to optimize prompts. Our framework addresses this limitation by analyzing textual gradients across the entire dataset to identify specific improvements. We evaluate its effectiveness within the medical domain through rigorous testing on established benchmarks such as MedQA, MedMCQA, PubMedQA, and the medical subset of MMLU.

## Methodology

The core of the project is **TextGrad**, a powerful package that optimizes multi-stage LLM agents by backpropagating textual gradients. This innovative approach allows for the systematic refinement of prompts, improving their effectiveness across a variety of tasks.

Key benchmarks for evaluation include:
- **MedQA**: USMLE-based medical question-answering dataset.
- **MedMCQA**: Indian postgraduate medical exam questions.
- **PubMedQA**: Questions derived from PubMed abstracts.
- **MMLU (Medical Subset)**: Nine clinically relevant topics in natural language understanding.

The framework is evaluated on both quantitative metrics (accuracy) and qualitative insights (clinical relevance of generated system prompts).

---

## Installation and Usage

### Prerequisites
Ensure you have Python installed, along with any required libraries. The `requirements.txt` file provided contains all necessary dependencies.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SeanWu25/med-auto-prompt.git
   cd med-auto-prompt
2. Install dependencies
   ```bash
     pip install -r requirements.txt
3. To actually run an experiment
    ```bash
    python main.py --benchmark="MedQA" --method="auto-prompt" --model="meta-llama/Llama-3-70b-chat-hf"

## Acknowledgments

This project would not have been possible without the **TextGrad** package, developed by the James Zou Group at Stanford University.
We leveraged TextGrad to systematically enhance LLM performance across medical benchmarks, facilitating meaningful comparisons with traditional methods.
For more information about TextGrad, visit their [GitHub repository](https://github.com/zou-group/textgrad) and [official documentation](https://textgrad.readthedocs.io/en/latest/).

We extend our gratitude to the creators of TextGrad for their groundbreaking work, which was integral to this research.


