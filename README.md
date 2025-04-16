# Text-to-SQL GRPO Fine-tuning Pipeline

This repository contains a pipeline for fine-tuning Large Language Models (LLMs) for Text-to-SQL conversion using General Reward Proximal Optimization (GRPO). The implementation focuses on Qwen2.5-Coder models but can be adapted for other LLMs.

## Overview

Text-to-SQL is the task of converting natural language questions into SQL queries. This project uses GRPO to fine-tune models, optimizing for:
- SQL correctness
- Clear reasoning
- Proper formatting
- Query complexity alignment

## Key Features

- **GRPO Fine-tuning**: Optimize models with multiple reward functions
- **Evaluation**: Comprehensive evaluation framework using gold queries and GPT-4o-mini
- **SQL Reward Functions**: Multiple reward metrics for SQL quality assessment
- **Contrastive Learning**: Improve natural language understanding for SQL generation

## Project Structure

- `llm_train.py`: Main training script for GRPO fine-tuning
- `sql_reward_utils.py`: SQL execution and reward functions
- `eval_grpo.py`: Evaluation of fine-tuned models
- `requirements.txt`: Required dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Clean the dataset:
```bash
python cleanse_dataset.py
```

This script filters the dataset to ensure:
- Valid SQL queries
- Correctly matched schema contexts
- Executable queries with proper syntax

## Training

Run the GRPO training:

```bash
python llm_train.py
```

Key parameters (can be modified in the script):
- `MODEL_NAME`: Base model to fine-tune (default: "Qwen/Qwen2.5-Coder-7B-Instruct")
- `MAX_SEQ_LENGTH`: Maximum sequence length (default: 1024)
- `LORA_RANK`: LoRA rank for parameter-efficient fine-tuning (default: 32)
- `BATCH_SIZE`: Training batch size (default: 4)
- `NUM_GENERATIONS`: Number of generations per prompt for GRPO (default: 8)
- `MAX_STEPS`: Maximum training steps (default: 225)

## Evaluation

Evaluate your trained model:

```bash
python eval_grpo.py
```

This script:
1. Loads your fine-tuned model
2. Generates SQL queries from test prompts
3. Evaluates the outputs using GPT-4o-mini
4. Produces detailed metrics and error analysis
5. Saves results as JSON and CSV

## Reward Functions

The training uses multiple reward components:

- **Format Reward**: Ensures proper XML tag structure
- **SQL Correctness**: Tests executable accuracy against gold standard
- **Complexity Reward**: Matches complexity between generated and gold queries
- **Reasoning Quality**: Assesses explanation quality and schema references

## Model Outputs

The model is trained to output in the following format:

```
<reasoning>
This database has a users table with columns for id, name, and age.
The question asks for all users over 30, so I need to query the users table with a WHERE condition.
</reasoning>
<sql>
SELECT * FROM users WHERE age > 30;
</sql>
```
