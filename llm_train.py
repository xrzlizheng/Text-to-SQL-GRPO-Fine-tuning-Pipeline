# 导入必要的库
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
    soft_format_reward_func,      # 软格式奖励函数
    strict_format_reward_func,    # 严格格式奖励函数
    execute_query_reward_func,    # SQL执行奖励函数
    complexity_reward,            # 复杂度奖励函数
    reasoning_quality_reward,     # 推理质量奖励函数
    REWARD_WEIGHTS               # 奖励权重配置
)

# 设置CUDA启动阻塞，确保GPU操作同步执行
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据文件路径配置
TRAIN_DATA_FILE = "cleaned_train_queries.jsonl"    # 训练数据文件
EVAL_DATA_FILE = "cleaned_eval_queries.jsonl"      # 评估数据文件

# 输出目录配置
OUTPUT_DIR = "outputs/sql_grpo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_gpu_memory(step=""):
    """
    打印当前GPU内存使用情况
    
    参数:
        step (str): 当前执行步骤的描述
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"\n--- GPU Memory at {step} ---")
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")
        print(f"Max memory allocated: {max_allocated:.2f} GB")


# 模型训练配置参数
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 基础模型名称
MAX_SEQ_LENGTH = 1024                          # 最大序列长度
LORA_RANK = 32                                # LoRA秩
BATCH_SIZE = 4                                # 批次大小
GRAD_ACCUMULATION = 2                         # 梯度累积步数
NUM_GENERATIONS = 8                           # 每个提示的生成数量
MAX_STEPS = 250                              # 最大训练步数
USE_WANDB = True                             # 是否使用Weights & Biases进行实验跟踪

# 数据集配置
DATASET_NAME = "gretelai/synthetic_text_to_sql"  # 数据集名称
NUM_EXAMPLES = 300                              # 使用的样本数量
DATASET_SPLIT = "train"                         # 数据集划分

# 奖励函数权重配置
REWARD_WEIGHTS = {
    "format": 1.0,              # 格式奖励权重
    "sql_correctness": 1.2,     # SQL正确性奖励权重
    "complexity": 0.6,          # 复杂度奖励权重
    "reasoning": 0.7,           # 推理质量奖励权重
}
SYNTAX_PENALTY = -0.1 * REWARD_WEIGHTS["sql_correctness"]  # SQL语法错误惩罚

# 配置Weights & Biases
if USE_WANDB:
    try:
        import wandb
    except ImportError:
        print("Wandb not installed. Disabling W&B logging.")
        USE_WANDB = False

# 打印训练配置信息
print("\n=== Starting SQL-to-Text Training Script ===")
print(f"Model: {MODEL_NAME}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"LoRA Rank: {LORA_RANK}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Gradient Accumulation: {GRAD_ACCUMULATION}")
print(f"Number of Generations: {NUM_GENERATIONS}")
print(f"Max Steps: {MAX_STEPS}")
print_gpu_memory("start")

# 系统提示模板
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
    """
    从文本中提取SQL查询语句
    
    参数:
        text (str): 包含SQL查询的文本
        
    返回:
        str: 提取出的SQL查询语句
    """
    if not text:
        return ""

    # 首先尝试从<sql>标签中提取
    match = re.search(r"<sql>(.*?)</sql>", text, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = re.sub(r"^\s*--.*?\n", "", sql)  # 移除开头的注释
        sql = re.sub(r"\n--.*?\s*$", "", sql)  # 移除结尾的注释
        return sql.strip()
    else:
        # 如果没有找到<sql>标签，尝试查找SQL关键字
        sql_keywords = ["SELECT ", "INSERT ", "UPDATE ", "DELETE ",
                        "CREATE ", "ALTER ", "DROP ", "TRUNCATE ",
                        "GRANT ", "REVOKE ", "MERGE ", "EXEC ", "WITH "]

        text_upper = text.upper()
        sql_start_index = -1
        keyword_found = ""

        # 查找第一个出现的SQL关键字
        for keyword in sql_keywords:
            idx = text_upper.find(keyword)
            if idx != -1:
                if sql_start_index == -1 or idx < sql_start_index:
                    sql_start_index = idx
                    keyword_found = keyword

        if sql_start_index != -1:
            potential_sql = text[sql_start_index:]
            # 移除推理部分
            if "</reasoning>" in potential_sql:
                potential_sql = potential_sql.split("</reasoning>", 1)[0]

            # 确保SQL语句以分号结束
            if ";" in potential_sql:
                potential_sql = potential_sql.split(";", 1)[0] + ";"
            return potential_sql.strip()

        return ""


def extract_schema_from_context(sql_context: str) -> str:
    """
    从SQL上下文中提取数据库模式信息
    
    参数:
        sql_context (str): 包含数据库模式的SQL上下文
        
    返回:
        str: 提取出的数据库模式信息
    """
    if not sql_context:
        return "No schema information available."
    statements = sqlparse.split(sql_context)
    # 只保留CREATE TABLE语句
    schema_statements = [
        s.strip() for s in statements
        if s.strip().upper().startswith("CREATE TABLE")
    ]
    extracted_schema = "\n".join(schema_statements)
    return extracted_schema if extracted_schema else sql_context


def filter_sql_context_for_training(sql_context: str) -> str:
    """
    过滤SQL上下文，只保留用于训练的模式信息
    
    参数:
        sql_context (str): 原始SQL上下文
        
    返回:
        str: 过滤后的SQL上下文
    """
    return extract_schema_from_context(sql_context)


try:
    logger.info("=== Loading Model ===")
    # 加载预训练模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,          # 使用4位量化加载模型
        fast_inference=True,        # 启用快速推理
        max_lora_rank=LORA_RANK,    # 设置LoRA最大秩
        dtype=None,                 # 自动选择数据类型
    )
    logger.info("Model loaded successfully")
    print_gpu_memory("after model load")

    logger.info("=== Applying LoRA ===")
    # 应用LoRA适配器到模型
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[            # 指定需要应用LoRA的模块
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK,       # LoRA缩放因子
        lora_dropout=0.05,          # LoRA dropout率
        bias="none",                # 不训练偏置项
        use_gradient_checkpointing="unsloth",  # 使用梯度检查点以节省显存
        random_state=3407,          # 随机种子
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
    # 加载训练数据集
    train_df = pd.read_json(TRAIN_DATA_FILE, lines=True)

    # 如果指定了样本数量，则随机采样
    if NUM_EXAMPLES and len(train_df) > NUM_EXAMPLES:
        dataset = train_df.sample(
            n=NUM_EXAMPLES, random_state=42).reset_index(drop=True)

    dataset = train_df.to_dict(orient='records')

    logger.info(f"Loaded {len(dataset)} examples from {DATASET_NAME}")
    print_gpu_memory("after dataset load")

    train_data = []
    logger.info("=== Preparing Dataset ===")

    # 处理每个训练样本
    for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
        sql_prompt = example.get("sql_prompt", "")      # 获取SQL提示
        sql_context = example.get("sql_context", "")    # 获取SQL上下文
        gold_sql = example.get("sql", "")              # 获取标准SQL查询

        # 检查数据完整性
        if not sql_prompt or not sql_context or not gold_sql:
            logger.warning(
                f"Skipping example {i} due to missing data (prompt, context, or gold SQL).")
            continue

        # 处理SQL上下文和模式
        filtered_context = filter_sql_context_for_training(sql_context)
        schema_for_prompt = extract_schema_from_context(filtered_context)

        # 构建聊天格式的提示
        prompt_chat = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"{schema_for_prompt}\n\nQuestion: {sql_prompt}"}
        ]
        prompt_string = tokenizer.apply_chat_template(
            prompt_chat, tokenize=False, add_generation_prompt=True
        )

        # 添加到训练数据列表
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
    """
    奖励日志回调类，用于记录训练过程中的奖励信息
    """
    def __init__(self):
        self.step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """
        在每个训练步骤结束时记录奖励信息
        
        参数:
            args: 训练参数
            state: 训练状态
            control: 训练控制
        """
        self.step += 1
        if self.step % 25 == 0:  # 每25步记录一次
            logger.info(f"\n--- Step {self.step} Reward Details (Sample) ---")
            if 'loss' in state.log_history[-1]:
                logger.info(
                    f" Step {self.step}: Current Loss: {state.log_history[-1]['loss']:.4f}")


def train_model():
    """
    训练模型的主函数
    """
    # 初始化Weights & Biases
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

    # 清理GPU缓存
    torch.cuda.empty_cache()
    print_gpu_memory("before trainer init")

    # 设置序列长度限制
    effective_max_completion_length = 300
    effective_max_prompt_length = MAX_SEQ_LENGTH - \
        effective_max_completion_length - 32

    # 配置训练参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,                    # 学习率
        per_device_train_batch_size=BATCH_SIZE,  # 每个设备的训练批次大小
        gradient_accumulation_steps=GRAD_ACCUMULATION,  # 梯度累积步数
        optim="adamw_8bit",                    # 优化器
        max_steps=MAX_STEPS,                   # 最大训练步数
        warmup_ratio=0.1,                      # 预热比例
        lr_scheduler_type="cosine",            # 学习率调度器类型
        logging_steps=5,                       # 日志记录步数
        save_steps=50,                         # 模型保存步数
        save_total_limit=2,                    # 保存的检查点数量限制
        save_strategy="steps",                 # 保存策略
        bf16=is_bfloat16_supported(),          # 是否使用bfloat16
        fp16=not is_bfloat16_supported(),      # 是否使用fp16
        gradient_checkpointing=True,           # 启用梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=effective_max_prompt_length,  # 最大提示长度
        max_completion_length=effective_max_completion_length,  # 最大完成长度
        num_generations=NUM_GENERATIONS,       # 生成数量
        beta=0.1,                              # GRPO beta参数
        use_vllm=True,                         # 使用vLLM加速
        report_to="wandb" if USE_WANDB else "none",  # 报告目标
        remove_unused_columns=False,           # 是否移除未使用的列
        seed=42,                               # 随机种子
        dataloader_num_workers=2,              # 数据加载器工作进程数
        max_grad_norm=1.0,                     # 梯度裁剪范数
    )

    logger.info("Initializing GRPOTrainer with improved reward functions...")
    # 初始化GRPO训练器
    trainer = GRPOTrainer(
        model=model,
        beta=training_args.beta,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        reward_funcs=[                          # 奖励函数列表
            soft_format_reward_func,            # 软格式奖励
            execute_query_reward_func,          # SQL执行奖励
            complexity_reward,                  # 复杂度奖励
            reasoning_quality_reward,           # 推理质量奖励
        ],
        callbacks=[RewardLoggerCallback()] if not USE_WANDB else None,
    )

    # 清理GPU缓存并开始训练
    torch.cuda.empty_cache()
    print_gpu_memory("before training starts")

    logger.info("Starting GRPO training...")
    try:
        trainer.train()  # 开始训练
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    # 保存最终的LoRA适配器
    final_save_path = f"{OUTPUT_DIR}/final_lora"
    logger.info(f"Saving final LoRA adapters to {final_save_path}...")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info("Model and tokenizer saved.")

    # 如果启用了WandB，保存最终模型并结束运行
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
    """
    使用样本查询测试训练好的模型
    
    参数:
        model: 训练好的模型
        tokenizer: 分词器
    """
    logger.info("\n=== Testing trained model with a sample query ===")

    EVAL_DATA_FILE = "cleaned_eval_queries.jsonl"

    try:
        # 加载评估数据集
        eval_df = pd.read_json(EVAL_DATA_FILE, lines=True)
        if eval_df.empty:
            raise ValueError(
                f"Evaluation dataset '{EVAL_DATA_FILE}' is empty.")

        # 随机选择一个评估样本
        eval_sample = eval_df.sample(n=1, random_state=123).iloc[0]

        sql_prompt = eval_sample.get("sql_prompt", "N/A")      # 获取SQL提示
        sql_context = eval_sample.get("sql_context", "")       # 获取SQL上下文
        gold_sql = eval_sample.get("sql", "N/A")              # 获取标准SQL查询

    except (ValueError, FileNotFoundError) as e:
        # 如果无法加载评估数据，使用默认样本
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

    # 准备测试提示
    schema_for_prompt = extract_schema_from_context(sql_context)
    test_prompt_chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{schema_for_prompt}\n\nQuestion: {sql_prompt}"}
    ]
    text = tokenizer.apply_chat_template(
        test_prompt_chat, tokenize=False, add_generation_prompt=True
    )

    # 将模型移至GPU并设置为评估模式
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # 清理GPU缓存
    torch.cuda.empty_cache()
    print_gpu_memory("before test generation")

    logger.info("Generating test response...")
    output_text = "[Generation Failed]"
    try:
        with torch.no_grad():  # 禁用梯度计算
            # 准备输入
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LENGTH).to(model.device)

            # 生成输出
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,      # 最大新生成token数
                temperature=0.2,         # 采样温度
                top_p=0.95,             # 核采样概率
                do_sample=True,         # 启用采样
                pad_token_id=tokenizer.eos_token_id  # 填充token ID
            )

            # 解码生成的token
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = output_ids[0][input_length:]
            output_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Error during test generation: {e}", exc_info=True)

    # 打印测试结果
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
    # 训练模型
    trained_model, trained_tokenizer = train_model()

    # 如果训练成功，进行测试
    if trained_model and trained_tokenizer:
        test_model(trained_model, trained_tokenizer)
    else:
        logger.error(
            "Training did not return a valid model or tokenizer. Skipping test.")

    logger.info("\nGRPO Training Script Completed.")
