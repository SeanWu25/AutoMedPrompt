# AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients


### 📘 Abstract
AutoMedPrompt is a novel framework that enhances the performance of large language models (LLMs) in medical question-answering tasks by utilizing **Textual Gradients**. Unlike traditional fine-tuning and manual prompt engineering, AutoMedPrompt dynamically optimizes prompts to elicit medically relevant reasoning using the **TextGrad** package's automatic differentiation capabilities.
<img width="966" alt="Screenshot 2025-02-22 at 10 26 46 PM" src="https://github.com/user-attachments/assets/2d61106e-87f7-442f-b8c5-bb5cef7c3d8c" />
AutoMedPrompt achieves state-of-the-art (SOTA) performance on multiple medical QA benchmarks including:
- **PubMedQA**: Accuracy of 82.6% (SOTA)
- **MedQA**: Accuracy of 77.7%
- **NephSAP**: Accuracy of 63.8%



### 🚀 Key Features
- **TextGrad Integration**: Leverages TextGrad's backpropagation mechanism for textual gradients.
- **Dynamic Prompt Optimization**: Automatically adjusts system prompts to align with medical reasoning and knowledge.
- **Outperforms Proprietary Models**: Surpasses GPT-4, Claude 3 Opus, and Med-PaLM 2 on multiple benchmarks.
- **Open-Source Excellence**: Achieves SOTA without any fine-tuning, using open-weight LLMs like **Llama 3**.


### 📊 Benchmark Performance


| **Prompting Strategy**            | **Model Size** | **Open-Weight** | **Avoids Finetuning** | **PubMedQA (%)** | **MedQA (%)** | **NephSAP (%)** |
|----------------------------------|----------------|-----------------|----------------------|------------------|---------------|-----------------|
| *Heuristic Baselines*             |                |                 |                      |                  |               |                 |
| Random Choice                     | -              | -               | -                    | 33.3             | 25.0          | 24.1            |
| Human Performance                 | -              | -               | -                    | 78.0             | 60.0          | 76.0            |
| *Large Language Models*           |                |                 |                      |                  |               |                 |
| GPT-4-base                        | ~1.76T         | ❌               | ✅                   | 80.4             | 81.7          | 63.6            |
| Claude 3 Opus                     | ~100B          | ❌               | ✅                   | 74.9             | 64.7          | 40.8            |
| Med-PaLM 2                        | ~1.6T          | ❌               | ❌                   | 81.8             | 85.4          | N/A             |
| MEDITRON                          | 70B            | ✅               | ❌                   | 80.0             | 65.4          | 29.8            |
| GPT-4 (Medprompt)                 | ~1.76T         | ❌               | ✅                   | 82.0             | 90.2          | N/A             |
| OpenMedLM                         | 34B            | ✅               | ✅                   | 77.3             | 72.6          | N/A             |
| Llama 3 (Zero-Shot)               | 70B            | ✅               | ✅                   | 71.6             | 75.2          | 61.6            |
| Llama 3 (Few-Shot)                | 70B            | ✅               | ✅                   | 76.0             | 75.4          | 58.8            |
| Llama 3 (DeepSeek R1 CoT)         | 70B            | ✅               | ✅                   | 71.4             | 76.4          | 48.0            |
| **Llama 3 (AutoMedPrompt)**       | **70B**        | **✅**           | **✅**               | **82.6**         | **77.7**      | **63.8**        |


### Notes:
- ✅ = Yes
- ❌ = No
- N/A = Not Applicable


### 🔬 Experiments

- Evaluated on open-source LLM **Llama 3 (70B)** and proprietary models:
  - **GPT-4**
  - **Claude 3 Opus**
  - **Med-PaLM 2**
  - **MEDITRON**

- Benchmarks used:
  - **MedQA**: USMLE-based multiple-choice QA
  - **PubMedQA**: Yes/No/Maybe QA based on PubMed abstracts
  - **NephSAP**: Nephrology-specific multiple-choice questions



### 🔥 Why AutoMedPrompt Outperforms
- **Task-Specific Prompting**: Tailors system prompts to unique medical tasks using textual gradients.
- **Dynamic Reasoning**: Outperforms static CoT and Few-Shot approaches by adapting to reasoning requirements in real-time.
- **No Fine-Tuning Needed**: Surpasses specialized models using generalist foundation models through advanced prompt optimization.

### 💻 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/SeanWu25/AutoMedPrompt.git

# Navigate to the project directory
cd AutoMedPrompt

# Install required packages
pip install -r requirements.txt

# Run the model
 python main.py --benchmark="MedQA" --method="auto-prompt" --model="meta-llama/Llama-3-70b-chat-hf"

# Evaluate an experiment
python eval.py --csv_folder="path to folder" --benchmark_name="NephSAP"
```


## 📜 Citations

If you use this work in your research, please cite the paper as follows:

```bibtex
@article{wu2025automedprompt,
  title={AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients},
  author={Wu, Sean and Koo, Michael and Scalzo, Fabien and Kurtz, Ira},
  journal={arXiv preprint arXiv:2502.15944},
  year={2025}
}
```

## 🙏 Acknowledgments
This work was built off of and only made possible because of **TextGrad**, a framework for optimizing complex multi-step LLM agents using textual gradients.

If you are interested in learning more about TextGrad, please refer to their paper:

```bibtex
@article{yuksekgonul2024textgrad,
  title={TextGrad: Automatic "Differentiation" via Text},
  author={Yuksekgonul, Mert and Bianchi, Federico and Boen, Joseph and Liu, Sheng and Huang, Zhi and Guestrin, Carlos and Zou, James},
  journal={arXiv preprint arXiv:2406.07496},
  year={2024}
}
```

