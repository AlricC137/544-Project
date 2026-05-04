# Detecting Hallucinations in LLMs Using Internal Activations and Knowledge-Grounded Reasoning

A layered hallucination detection and correction framework for open-source LLMs, combining **LoRA fine-tuning**, **Self-Consistency decoding**, and a **zero-shot Generator-Checker (G-C)** framework — evaluated on the HaluEval benchmark across QA, dialogue, and summarization tasks.

> **Models:** Qwen2.5-3B-Instruct · Gemma-3-4B-IT  
> **Benchmark:** [HaluEval](https://arxiv.org/abs/2305.11747)  (QA, Dialogue, Summarization)  
> **Fine-tuning dataset:** [TruthfulQA](https://arxiv.org/abs/2109.07958)

---

## Table of Contents

- [Detecting Hallucinations in LLMs Using Internal Activations and Knowledge-Grounded Reasoning](#detecting-hallucinations-in-llms-using-internal-activations-and-knowledge-grounded-reasoning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Reproducing Results](#reproducing-results)
    - [1. Data Preparation](#1-data-preparation)
    - [2. LoRA Fine-Tuning](#2-lora-fine-tuning)
    - [3. Baseline \& LoRA Evaluation](#3-baseline--lora-evaluation)
    - [4. Self-Consistency Evaluation](#4-self-consistency-evaluation)
    - [5. Generator-Checker Evaluation](#5-generator-checker-evaluation)
  - [Results](#results)
    - [Hallucination Detection](#hallucination-detection)
      - [Question Answering](#question-answering)
      - [Dialogue](#dialogue)
      - [Summarization](#summarization)
    - [Generator-Checker Framework](#generator-checker-framework)
      - [Question Answering](#question-answering-1)
      - [Dialogue](#dialogue-1)
      - [Summarization](#summarization-1)
  - [Authors](#authors)

---

## Overview

Large Language Models (LLMs) frequently produce fluent but factually incorrect outputs — a phenomenon known as *hallucination*. This project investigates a layered mitigation pipeline:

1. **LoRA Fine-Tuning** — Parameter-efficient fine-tuning on TruthfulQA to promote factually grounded outputs.
2. **Self-Consistency (SC)** — Generates 3 candidate outputs per sample and aggregates via majority voting, with lexical and semantic divergence tracking to flag uncertain predictions.
3. **Generator-Checker Framework** — A zero-shot closed-loop setup where one model generates a response and a second model independently verifies and provides corrective feedback, allowing one round of refinement.

---

## Project Structure

```
.
├── data/
│   ├── test/
│   │   ├── instruction_files/
│   │   │   ├── dialogue_evaluation_instruction.txt
│   │   │   ├── qa_evaluation_instruction.txt
│   │   │   └── summarization_evaluation_instruction.txt
│   │   ├── halueval_dialogue_data.xlsx
│   │   ├── halueval_qa_data.xlsx
│   │   └── halueval_summarization_data.xlsx
│   └── train/
│       └── truthfulqa_generation_data.xlsx
│
├── train/
│   ├── gemma/
│   │   ├── model_files/
│   │   │   ├── model/
│   │   │   │   ├── adapter_config.json
│   │   │   │   └── adapter_model.safetensors
│   │   │   └── tokenizer/
│   │   │       ├── chat_template.jinja
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   └── lora_finetuning_gemma.ipynb
│   └── qwen/
│       ├── model_files/
│       │   ├── model/
│       │   │   ├── adapter_config.json
│       │   │   └── adapter_model.safetensors
│       │   └── tokenizer/
│       │       ├── chat_template.jinja
│       │       ├── tokenizer_config.json
│       │       └── tokenizer.json
│       └── lora_finetuning_qwen.ipynb
│
├── evaluation/
│   ├── benchmark_lora_eval.ipynb
│   ├── generator_checker_eval.ipynb
│   └── self_consistency_eval.ipynb
│
├── test_results/
│   ├── gemma/                          # Per-task JSONL results for Gemma
│   ├── qwen/                           # Per-task JSONL results for Qwen
│   ├── generator_checker/              # Cross-model G-C JSONL results
│   ├── results_benchmark_lora_selfconsistency.xlsx
│   └── results_generator_checker.xlsx
│
├── README.md
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended: ≥32GB VRAM for fine-tuning)
- Jupyter Notebook or JupyterLab

> **Note:** This project was developed and tested on [Google Colab](https://colab.research.google.com/) using an **H100 GPU**. We recommend running the notebooks in the same environment for best compatibility and performance.

### Installation

```bash
git clone https://github.com/AlricC137/544-Project.git
cd 544-Project
pip install -r requirements.txt
```

The `requirements.txt` covers all dependencies including and related libraries.

---

## Reproducing Results

### 1. Data Preparation

The training and evaluation data are pre-processed and available under `data/`. No additional download is required for TruthfulQA or HaluEval.

- **Training data:** `data/train/truthfulqa_generation_data.xlsx`  
  Contains 8,514 question–answer pairs from TruthfulQA (correct answers only).

- **Evaluation data:** `data/test/halueval_{qa,dialogue,summarization}_data.xlsx`  
  Each file contains 10,000 balanced samples (hallucinated vs. factual) from HaluEval.

- **Instruction prompts:** `data/test/instruction_files/`  
  Task-specific system instructions appended to each evaluation prompt.

---

### 2. LoRA Fine-Tuning

Run the fine-tuning notebooks for each model independently. Pre-trained LoRA adapter weights are already saved under `train/{gemma,qwen}/model_files/` if you wish to skip this step.

**Fine-tune Gemma:**
```
train/gemma/lora_finetuning_gemma.ipynb
```

**Fine-tune Qwen:**
```
train/qwen/lora_finetuning_qwen.ipynb
```

After training, LoRA weights are merged into the base model for inference.

---

### 3. Baseline & LoRA Evaluation

Run the benchmark evaluation notebook to evaluate both models in **Base** and **LoRA** configurations across all three HaluEval tasks:

```
evaluation/benchmark_lora_eval.ipynb
```

This notebook:
- Constructs task-specific prompts using the instruction files
- Runs greedy decoding
- Parses "Yes"/"No" judgments
- Outputs per-task metrics (accuracy, precision, recall, f1 score)
- Saves results to `test_results/{gemma,qwen}/`
- Aggregates all results to `test_results/results_benchmark_lora_selfconsistency.xlsx`

---

### 4. Self-Consistency Evaluation

Run the self-consistency notebook on top of the LoRA fine-tuned models:

```
evaluation/self_consistency_eval.ipynb
```

This notebook:
- Generates 3 stochastic outputs per sample
- Aggregates predictions via majority voting
- Computes lexical divergence (token-level F1) and semantic divergence (MiniLM cosine similarity)
- Flags samples with combined divergence > 0.35 as uncertain
- Saves SC results to `test_results/{gemma,qwen}/`
- Aggregates all results to `test_results/results_benchmark_lora_selfconsistency.xlsx`

---

### 5. Generator-Checker Evaluation

Run the Generator-Checker evaluation notebook using the **baseline** (non-fine-tuned) versions of both models:

```
evaluation/generator_checker_eval.ipynb
```

This notebook evaluates two cross-model pairings:
- **Gemma (generator) → Qwen (checker)**
- **Qwen (generator) → Gemma (checker)**

For each sample:
1. The checker judges the generated response ("Yes"/"No" + optional feedback)
2. If "Yes" (hallucinated), feedback is passed to the generator for one round of revision
3. The checker re-evaluates the revised response (final verdict)

Generator: temperature=0.7, top-p=0.9, max 256 tokens  
Checker: greedy decoding, max 128 tokens

Results saved to `test_results/generator_checker/` and summarized in `test_results/results_generator_checker.xlsx`.

---

## Results
 
### Hallucination Detection
 
#### Question Answering
 
| Model | Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Qwen | Base | 0.4962 | 0.4000 | 0.0112 | 0.0217 |
| Qwen | LoRA | 0.5022 | 0.5392 | 0.0439 | 0.0812 |
| Qwen | SC-LoRA | 0.5026 | 0.5217 | 0.0864 | 0.1483 |
| Gemma | Base | 0.5012 | 0.5013 | 0.8723 | 0.6367 |
| Gemma | LoRA | 0.5012 | 0.5293 | 0.0397 | 0.0739 |
| Gemma | SC-LoRA | 0.4943 | 0.4901 | 0.2323 | 0.3152 |
 
#### Dialogue
 
| Model | Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Qwen | Base | 0.5869 | 0.7039 | 0.3028 | 0.4234 |
| Qwen | LoRA | 0.5870 | 0.8250 | 0.2230 | 0.3510 |
| Qwen | SC-LoRA | 0.5892 | 0.7194 | 0.2952 | 0.4186 |
| Gemma | Base | 0.6125 | 0.5890 | 0.7497 | 0.6597 |
| Gemma | LoRA | 0.5204 | 0.8567 | 0.0513 | 0.0968 |
| Gemma | SC-LoRA | 0.5333 | 0.5729 | 0.2691 | 0.3662 |
 
#### Summarization
 
| Model | Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Qwen | Base | 0.2975 | 0.4240 | 0.4944 | 0.4565 |
| Qwen | LoRA | 0.3629 | 0.7000 | 0.0161 | 0.0315 |
| Qwen | SC-LoRA | 0.4610 | 0.5729 | 0.2691 | 0.3662 |
| Gemma | Base | 0.3869 | 0.5252 | 0.7567 | 0.6200 |
| Gemma | LoRA | 0.3625 | 0.0000 | 0.0000 | 0.0000 |
| Gemma | SC-LoRA | 0.4876 | 0.4898 | 0.4876 | 0.3841 |
 
### Generator-Checker Framework
 
#### Question Answering
 
| Generator | Checker | Avg Rounds | Detection Rate | Correction Rate |
|---|---|---|---|---|
| Gemma | Qwen | 1.84 | 0.7759 | 0.3150 |
| Qwen | Gemma | 1.00 | 0.0000 | 0.0000 |
 
#### Dialogue
 
| Generator | Checker | Avg Rounds | Detection Rate | Correction Rate |
|---|---|---|---|---|
| Gemma | Qwen | 1.12 | 0.1950 | 0.9104 |
| Qwen | Gemma | 1.00 | 0.0078 | 0.1250 |
 
#### Summarization
 
| Generator | Checker | Avg Rounds | Detection Rate | Correction Rate |
|---|---|---|---|---|
| Gemma | Qwen | 1.37 | 0.5112 | 0.7400 |
| Qwen | Gemma | 1.00 | 0.0019 | 0.0000 |
 
**Key findings:**
- LoRA fine-tuning disrupts base model detection behavior across tasks, especially for Gemma
- Self-Consistency is the most effective recovery mechanism, with the largest gains in summarization
- The Generator-Checker framework is highly asymmetric — Gemma→Qwen substantially outperforms the reverse across all tasks
  
---

## Authors

Pranay Obla Anandbabu · Tavion Fernandes · Ojas Golatkar · Kaustubh Mhatre · Anchalaa Jha

---