import json
import os
import re
import sqlite3
import tempfile
from tqdm import tqdm
import logging

import pandas as pd
import sqlparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from sql_reward_utils import (
    soft_format_reward_func,
    strict_format_reward_func,
    execute_query_reward_func,
    complexity_reward,
    reasoning_quality_reward,
    REWARD_WEIGHTS
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAIN_DATA_FILE = "cleaned_train_queries.jsonl"
EVAL_DATA_FILE = "cleaned_eval_queries.jsonl"

OUTPUT_DIR = "outputs/sql_grpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_gpu_memory(step=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n--- GPU Memory at {step} ---")
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")
        print(f"Max memory allocated: {max_allocated:.2f} GB")


MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2
NUM_GENERATIONS = 8
MAX_STEPS = 250
USE_WANDB = True

DATASET_NAME = "gretelai/synthetic_text_to_sql"
NUM_EXAMPLES = 300
DATASET_SPLIT = "train"

REWARD_WEIGHTS = {
    "format": 1.0,
    "sql_correctness": 1.2,
    "complexity": 0.6,
    "reasoning": 0.7,
}
SYNTAX_PENALTY = -0.1 * REWARD_WEIGHTS["sql_correctness"]

if USE_WANDB:
    try:
        import wandb
    except ImportError:
        print("Wandb not installed. Disabling W&B logging.")
        USE_WANDB = False

print("\n=== Starting SQL-to-Text Training Script ===")
print(f"Model: {MODEL_NAME}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"LoRA Rank: {LORA_RANK}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRAD_ACCUMULATION}")
print(f"Number of Generations: {NUM_GENERATIONS}")
print(f"Max Steps: {MAX_STEPS}")
print_gpu_memory("start")

SYSTEM_PROMPT = """
You are an AI assistant that converts natural language questions into SQL queries compatible with PostgreSQL syntax.
Given a database schema and a question, generate the correct PostgreSQL query.

Think about the problem and provide your working out.
Place it between <reasoning> and </reasoning>.
Then, provide your solution between <sql> and </sql>.

Here's an example of how you should respond:

<reasoning>
This database has a users table with columns for id, name, and age.
The question asks for all users over 30, so I need to query the users table with a WHERE condition.
</reasoning>
<sql>
SELECT * FROM users WHERE age > 30;
</sql>

Respond ONLY in the format above, including the <reasoning> and <sql> tags.
"""


def extract_sql(text: str) -> str:
    if not text:
        return ""

    match = re.search(r"<sql>(.*?)</sql>", text, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = re.sub(r"^\s*--.*?\n", "", sql)
        sql = re.sub(r"\n--.*?\s*$", "", sql)
        return sql.strip()
    else:
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ",
                        "CREATE ", "ALTER ", "DROP ", "TRUNCATE ",
                        "GRANT ", "REVOKE ", "MERGE ", "EXEC ", "WITH "]

        text_upper = text.upper()
        sql_start_index = -1
        keyword_found = ""

        for keyword in sql_keywords:
            idx = text_upper.find(keyword)
            if idx != -1:
                if sql_start_index == -1 or idx < sql_start_index:
                    sql_start_index = idx
                    keyword_found = keyword

        if sql_start_index != -1:
            potential_sql = text[sql_start_index:]
            if "</reasoning>" in potential_sql:
                potential_sql = potential_sql.split("</reasoning>", 1)[0]

            if ";" in potential_sql:
                potential_sql = potential_sql.split(";", 1)[0] + ";"
            return potential_sql.strip()

        return ""


def extract_schema_from_context(sql_context: str) -> str:
    if not sql_context:
        return "No schema information available."
    statements = sqlparse.split(sql_context)
    schema_statements = [
        s.strip() for s in statements
        if s.strip().upper().startswith("CREATE TABLE")
    ]
    extracted_schema = "\n".join(schema_statements)
    return extracted_schema if extracted_schema else sql_context


def filter_sql_context_for_training(sql_context: str) -> str:
    return extract_schema_from_context(sql_context)


try:
    logger.info("=== Loading Model ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        dtype=None,
    )
    logger.info("Model loaded successfully")
    print_gpu_memory("after model load")

    logger.info("=== Applying LoRA ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,
        lora_dropout=0.05,  # 0.1
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    logger.info("LoRA adapters applied successfully")
    print_gpu_memory("after LoRA")
except Exception as e:
    logger.error(
        f"Error in model loading or LoRA application: {e}", exc_info=True)
    exit(1)

try:
    logger.info("=== Loading Dataset ===")
    train_df = pd.read_json(TRAIN_DATA_FILE, lines=True)

    if NUM_EXAMPLES and len(train_df) > NUM_EXAMPLES:
        dataset = train_df.sample(
            n=NUM_EXAMPLES, random_state=42).reset_index(drop=True)

    dataset = train_df.to_dict(orient='records')

    logger.info(f"Loaded {len(dataset)} examples from {DATASET_NAME}")
    print_gpu_memory("after dataset load")

    train_data = []
    logger.info("=== Preparing Dataset ===")

    for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
        sql_prompt = example.get("sql_prompt", "")
        sql_context = example.get("sql_context", "")
        gold_sql = example.get("sql", "")

        if not sql_prompt or not sql_context or not gold_sql:
            logger.warning(
                f"Skipping example {i} due to missing data (prompt, context, or gold SQL).")
            continue

        filtered_context = filter_sql_context_for_training(sql_context)
        schema_for_prompt = extract_schema_from_context(filtered_context)

        prompt_chat = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"{schema_for_prompt}\n\nQuestion: {sql_prompt}"}
        ]
        prompt_string = tokenizer.apply_chat_template(
            prompt_chat, tokenize=False, add_generation_prompt=True
        )

        train_data.append({
            'prompt': prompt_string,
            'references': [{
                'gold_sql': gold_sql,
                'sql_context': sql_context,
                'sql_prompt': sql_prompt
            }],
        })

    logger.info(f"Prepared {len(train_data)} training examples")
    print_gpu_memory("after data preparation")

    if not train_data:
        logger.error(
            "No valid training data could be prepared. Check dataset format and content.")
        exit(1)

except Exception as e:
    logger.error(f"Error in data preparation: {e}", exc_info=True)
    exit(1)


class RewardLoggerCallback(TrainerCallback):
    def __init__(self):
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % 25 == 0:
            logger.info(f"\n--- Step {self.step} Reward Details (Sample) ---")
            if 'loss' in state.log_history[-1]:
                logger.info(
                    f" Step {self.step}: Current Loss: {state.log_history[-1]['loss']:.4f}")


def train_model():
    if USE_WANDB:
        try:
            if wandb.run is None:
                wandb.init(
                    project="text-to-sql-finetuning",
                    name=f"sql-grpo-{MODEL_NAME.split('/')[-1]}-{MAX_STEPS}steps",
                    config={
                        "model_name": MODEL_NAME,
                        "lora_rank": LORA_RANK,
                        "max_seq_length": MAX_SEQ_LENGTH,
                        "batch_size": BATCH_SIZE,
                        "grad_accumulation": GRAD_ACCUMULATION,
                        "num_generations": NUM_GENERATIONS,
                        "max_steps": MAX_STEPS,
                        "dataset": DATASET_NAME,
                        "num_examples": NUM_EXAMPLES,
                        "learning_rate": 5e-6,
                        "weight_decay": 0.01,
                        "warmup_ratio": 0.1,
                        "lr_scheduler_type": "cosine",
                        "optim": "adamw_8bit",
                        "syntax_penalty": SYNTAX_PENALTY,
                        "reward_weights": REWARD_WEIGHTS,
                        "stage": "grpo",
                    },
                    resume="allow",
                    save_code=True,
                )
            else:
                logger.info("WandB already initialized, resuming run.")
        except Exception as e:
            logger.error(f"WandB initialization failed: {e}", exc_info=True)

    torch.cuda.empty_cache()
    print_gpu_memory("before trainer init")

    effective_max_completion_length = 300
    effective_max_prompt_length = MAX_SEQ_LENGTH - \
        effective_max_completion_length - 32

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        optim="adamw_8bit",
        max_steps=MAX_STEPS,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        save_strategy="steps",
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=effective_max_prompt_length,
        max_completion_length=effective_max_completion_length,
        num_generations=NUM_GENERATIONS,
        beta=0.1,
        use_vllm=True,
        report_to="wandb" if USE_WANDB else "none",
        remove_unused_columns=False,
        seed=42,
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )

    logger.info("Initializing GRPOTrainer with improved reward functions...")
    trainer = GRPOTrainer(
        model=model,
        beta=training_args.beta,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=[
            soft_format_reward_func,
            execute_query_reward_func,
            complexity_reward,
            reasoning_quality_reward,
        ],
        callbacks=[RewardLoggerCallback()] if not USE_WANDB else None,
    )

    torch.cuda.empty_cache()
    print_gpu_memory("before training starts")

    logger.info("Starting GRPO training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    final_save_path = f"{OUTPUT_DIR}/final_lora"
    logger.info(f"Saving final LoRA adapters to {final_save_path}...")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info("Model and tokenizer saved.")

    if USE_WANDB and wandb.run:
        try:
            logger.info("Logging final model artifacts to WandB...")
            wandb.save(f"{final_save_path}/*")
            wandb.finish()
            logger.info("WandB run finished.")
        except Exception as e:
            logger.error(
                f"Failed to finish WandB run or save artifacts: {e}", exc_info=True)

    print_gpu_memory("after training")
    return model, tokenizer


def test_model(model, tokenizer):
    logger.info("\n=== Testing trained model with a sample query ===")

    EVAL_DATA_FILE = "cleaned_eval_queries.jsonl"

    try:
        eval_df = pd.read_json(EVAL_DATA_FILE, lines=True)
        if eval_df.empty:
            raise ValueError(
                f"Evaluation dataset '{EVAL_DATA_FILE}' is empty.")

        eval_sample = eval_df.sample(n=1, random_state=123).iloc[0]

        sql_prompt = eval_sample.get("sql_prompt", "N/A")
        sql_context = eval_sample.get("sql_context", "")
        gold_sql = eval_sample.get("sql", "N/A")

    except (ValueError, FileNotFoundError) as e:
        logger.warning(
            f"Could not load eval sample: {e}. Using a default sample.")
        sql_prompt = "List the names of departments with more than 10 employees."
        sql_context = """
        CREATE TABLE departments (department_id INT PRIMARY KEY, name TEXT);
        CREATE TABLE employees (employee_id INT PRIMARY KEY, name TEXT, department_id INT, FOREIGN KEY (department_id) REFERENCES departments(department_id));
        """
        gold_sql = """
        SELECT T1.name FROM departments AS T1 JOIN employees AS T2 ON T1.department_id = T2.department_id GROUP BY T1.department_id HAVING count(*) > 10
        """

    schema_for_prompt = extract_schema_from_context(sql_context)
    test_prompt_chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{schema_for_prompt}\n\nQuestion: {sql_prompt}"}
    ]
    text = tokenizer.apply_chat_template(
        test_prompt_chat, tokenize=False, add_generation_prompt=True
    )

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    torch.cuda.empty_cache()
    print_gpu_memory("before test generation")

    logger.info("Generating test response...")
    output_text = "[Generation Failed]"
    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LENGTH).to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = output_ids[0][input_length:]
            output_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error during test generation: {e}", exc_info=True)

    print("\n--- Test Results ---")
    print(f"Question: {sql_prompt}")
    print("-" * 40)
    print(f"Gold SQL:\n{gold_sql}")
    print("-" * 40)
    generated_sql = extract_sql(output_text)
    print(
        f"Generated SQL:\n{generated_sql if generated_sql else '[No SQL Extracted]'}")
    print("-" * 40)
    print(f"Full Generated Output:\n{output_text}")
    print("-" * 40)

    print_gpu_memory("after test")


if __name__ == "__main__":
    trained_model, trained_tokenizer = train_model()

    if trained_model and trained_tokenizer:
        test_model(trained_model, trained_tokenizer)
    else:
        logger.error(
            "Training did not return a valid model or tokenizer. Skipping test.")

    logger.info("\nGRPO Training Script Completed.")
