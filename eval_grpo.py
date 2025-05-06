"""
Text-to-SQL评估脚本
用于评估模型生成的SQL查询的质量和正确性
主要功能包括：
1. 加载预训练模型
2. 生成SQL查询
3. 使用GPT-4o-mini进行自动评估
4. 计算各种评估指标
5. 生成评估报告
"""

import json
import re
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from openai import OpenAI
from unsloth import FastLanguageModel
from peft import PeftModel

# 评估配置参数
EVAL_FILE = "cleaned_eval_queries.jsonl"  # 评估数据集文件
NUM_SAMPLES = 50                          # 评估样本数量
OPENAI_API_KEY = ''                       # OpenAI API密钥
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 使用的模型名称
FINETUNED_PATH = "outputs/sql_grpo/final_lora"  # 微调模型路径
MAX_SEQ_LENGTH = 1024                     # 最大序列长度
RESULT_FILE = "evaluation_results.json"   # 评估结果输出文件
CONCURRENT_REQUESTS = 5                   # 并发请求数

# 系统提示模板
SYSTEM_PROMPT = """
You are an AI assistant that converts natural language questions into SQL queries compatible with PostgreSQL syntax.
Given a database schema and a question, generate the correct PostgreSQL query.

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

def extract_sql(text):
    """
    从文本中提取SQL查询语句
    
    参数:
        text: 包含SQL查询的文本
        
    返回:
        str: 提取出的SQL查询语句
    """
    match = re.search(r"<sql>(.*?)</sql>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_reasoning(text):
    """
    从文本中提取推理部分
    
    参数:
        text: 包含推理的文本
        
    返回:
        str: 提取出的推理文本
    """
    match = re.search(r"<reasoning>(.*?)</reasoning>",
                      text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_gpt_evaluation(evaluation_text):
    """
    解析GPT评估结果
    
    参数:
        evaluation_text: GPT生成的评估文本
        
    返回:
        dict: 包含各项评估指标的字典
    """
    result = {
        "SQL_SCORE": 0,           # SQL正确性得分
        "REASONING_SCORE": 0,     # 推理质量得分
        "FORMAT_SCORE": 0,        # 格式正确性得分
        "EDUCATIONAL_SCORE": 0,   # 教育价值得分
        "OVERALL_SCORE": 0,       # 总体得分
        "EXPLANATION": "",        # 评估解释
        "ERROR_TYPE": "unknown"   # 错误类型
    }

    # 使用正则表达式提取各项得分
    sql_score = re.search(r"SQL_SCORE:\s*(\d+)", evaluation_text)
    reasoning_score = re.search(r"REASONING_SCORE:\s*(\d+)", evaluation_text)
    format_score = re.search(r"FORMAT_SCORE:\s*(\d+)", evaluation_text)
    educational_score = re.search(
        r"EDUCATIONAL_SCORE:\s*(\d+)", evaluation_text)
    overall_score = re.search(r"OVERALL_SCORE:\s*(\d+\.?\d*)", evaluation_text)
    error_type = re.search(r"ERROR_TYPE:\s*([^\n]+)", evaluation_text)

    explanation_match = re.search(
        r"EXPLANATION:\s*(.*?)(?=ERROR_TYPE:|$)", evaluation_text, re.DOTALL)

    # 更新结果字典
    if sql_score:
        result["SQL_SCORE"] = int(sql_score.group(1))
    if reasoning_score:
        result["REASONING_SCORE"] = int(reasoning_score.group(1))
    if format_score:
        result["FORMAT_SCORE"] = int(format_score.group(1))
    if educational_score:
        result["EDUCATIONAL_SCORE"] = int(educational_score.group(1))
    if overall_score:
        result["OVERALL_SCORE"] = float(overall_score.group(1))
    if explanation_match:
        result["EXPLANATION"] = explanation_match.group(1).strip()
    if error_type:
        result["ERROR_TYPE"] = error_type.group(1).strip()

    return result

def load_model():
    """
    加载预训练模型
    
    返回:
        tuple: (模型, 分词器)
    """
    print("Loading model...")
    try:
        # 加载基础模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            device_map="auto",
        )

        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)
        model.eval()

        print(f"Model loaded successfully from {FINETUNED_PATH}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def generate_response(model, tokenizer, prompt):
    """
    使用模型生成响应
    
    参数:
        model: 预训练模型
        tokenizer: 分词器
        prompt: 输入提示
        
    返回:
        str: 生成的响应文本
    """
    try:
        # 构建聊天格式的提示
        prompt_chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            prompt_chat, tokenize=False, add_generation_prompt=True)

        # 准备输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LENGTH).to(model.device)

        # 生成响应
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码生成的文本
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = output_ids[0][input_length:]
        output_text = tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return output_text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "[Error generating response]"

def evaluate_with_gpt4o_mini(samples, client):
    """
    使用GPT-4o-mini评估生成的SQL查询
    
    参数:
        samples: 评估样本列表
        client: OpenAI客户端实例
        
    返回:
        list: 评估结果列表
    """
    results = []

    for sample in tqdm(samples, desc="Evaluating with GPT-4o-mini"):
        # 构建评估提示
        eval_prompt = f"""
        As an SQL expert, evaluate this text-to-SQL conversion. Score each dimension from 1-5 (1=poor, 5=excellent).

        DATABASE SCHEMA:
        {sample['sql_context']}

        QUESTION:
        {sample['sql_prompt']}

        GOLD SQL (CORRECT):
        {sample['sql']}

        MODEL OUTPUT:
        {sample['model_output']}

        Provide scores in this exact format:
        SQL_SCORE: [1-5] - Does the SQL work and produce correct results?
        REASONING_SCORE: [1-5] - Is the reasoning clear, logical, and references correct schema?
        FORMAT_SCORE: [1-5] - Does it follow <reasoning>...</reasoning><sql>...</sql> format?
        EDUCATIONAL_SCORE: [1-5] - Would this help someone learn SQL?
        OVERALL_SCORE: [average]
        EXPLANATION: [brief explanation of strengths/weaknesses]
        ERROR_TYPE: [none/syntax/logic/format/other]
        """

        try:
            # 调用GPT-4o-mini进行评估
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1
            )

            evaluation = response.choices[0].message.content
            eval_result = parse_gpt_evaluation(evaluation)

            # 构建结果字典
            result = {
                "sample_id": sample.get("id", len(results)),
                "question": sample["sql_prompt"],
                "sql_context": sample["sql_context"],
                "gold_sql": sample["sql"],
                "model_output": sample["model_output"],
                "extracted_sql": sample["extracted_sql"],
                "extracted_reasoning": sample["extracted_reasoning"],
                "evaluation": eval_result,
                "raw_evaluation": evaluation
            }

            results.append(result)
            time.sleep(0.5)  # 避免API限制

        except Exception as e:
            print(f"Error evaluating sample: {e}")
            results.append({
                "sample_id": sample.get("id", len(results)),
                "question": sample["sql_prompt"],
                "sql_context": sample["sql_context"],
                "gold_sql": sample["sql"],
                "model_output": sample["model_output"],
                "extracted_sql": sample["extracted_sql"],
                "extracted_reasoning": sample["extracted_reasoning"],
                "evaluation": {"ERROR": str(e)},
                "evaluation_failed": True
            })

    return results

def format_results_summary(evaluation_results):
    """
    格式化评估结果摘要
    
    参数:
        evaluation_results: 评估结果列表
        
    返回:
        str: 格式化的评估摘要
    """
    # 过滤出有效的评估结果
    valid_results = [r for r in evaluation_results if not r.get(
        "evaluation_failed", False)]

    if not valid_results:
        return "No valid evaluation results."

    # 收集各项得分
    scores = {
        "SQL_SCORE": [r["evaluation"]["SQL_SCORE"] for r in valid_results],
        "REASONING_SCORE": [r["evaluation"]["REASONING_SCORE"] for r in valid_results],
        "FORMAT_SCORE": [r["evaluation"]["FORMAT_SCORE"] for r in valid_results],
        "EDUCATIONAL_SCORE": [r["evaluation"]["EDUCATIONAL_SCORE"] for r in valid_results],
        "OVERALL_SCORE": [r["evaluation"]["OVERALL_SCORE"] for r in valid_results]
    }

    # 统计错误类型
    error_types = Counter([r["evaluation"]["ERROR_TYPE"]
                          for r in valid_results])

    # 构建摘要
    summary = "=== EVALUATION SUMMARY ===\n\n"

    # 添加平均得分
    summary += "AVERAGE SCORES:\n"
    for metric, values in scores.items():
        summary += f"  {metric}: {np.mean(values):.2f} (±{np.std(values):.2f})\n"

    # 添加得分分布
    summary += "\nSCORE DISTRIBUTION:\n"
    for metric, values in scores.items():
        counts = Counter(values)
        summary += f"  {metric}: " + " | ".join(
            [f"{score}={count}" for score, count in sorted(counts.items())]) + "\n"

    # 添加错误类型统计
    summary += "\nERROR TYPES:\n"
    for error_type, count in error_types.most_common():
        summary += f"  {error_type}: {count} ({count/len(valid_results)*100:.1f}%)\n"

    # 添加样本统计
    summary += f"\nTotal samples evaluated: {len(evaluation_results)}\n"
    summary += f"Valid evaluations: {len(valid_results)}\n"

    return summary

def main():
    """
    主函数：执行评估流程
    """
    # 检查API密钥
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY environment variable not set. Please set it before running this script.")
        return

    # 初始化OpenAI客户端
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 加载模型
    model, tokenizer = load_model()

    # 加载评估数据
    print(f"Loading evaluation data from {EVAL_FILE}...")
    try:
        with open(EVAL_FILE, 'r') as f:
            eval_data = [json.loads(line) for line in f]

        eval_subset = eval_data[:NUM_SAMPLES]
        print(f"Loaded {len(eval_subset)} samples for evaluation")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    # 生成模型输出
    print("Generating model outputs...")
    for sample in tqdm(eval_subset, desc="Generating responses"):
        prompt = f"Database schema:\n{sample['sql_context']}\n\nQuestion: {sample['sql_prompt']}"
        sample["model_output"] = generate_response(model, tokenizer, prompt)
        sample["extracted_sql"] = extract_sql(sample["model_output"])
        sample["extracted_reasoning"] = extract_reasoning(
            sample["model_output"])

    # 使用GPT-4o-mini进行评估
    print("Evaluating with GPT-4o-mini...")
    evaluation_results = evaluate_with_gpt4o_mini(eval_subset, client)

    # 保存评估结果
    print(f"Saving results to {RESULT_FILE}...")
    with open(RESULT_FILE, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    # 生成CSV格式的评估摘要
    csv_data = []
    for result in evaluation_results:
        if result.get("evaluation_failed", False):
            continue

        csv_data.append({
            "id": result["sample_id"],
            "question": result["question"],
            "sql_score": result["evaluation"]["SQL_SCORE"],
            "reasoning_score": result["evaluation"]["REASONING_SCORE"],
            "format_score": result["evaluation"]["FORMAT_SCORE"],
            "educational_score": result["evaluation"]["EDUCATIONAL_SCORE"],
            "overall_score": result["evaluation"]["OVERALL_SCORE"],
            "error_type": result["evaluation"]["ERROR_TYPE"]
        })

    pd.DataFrame(csv_data).to_csv("evaluation_summary.csv", index=False)

    # 生成文本格式的评估摘要
    summary = format_results_summary(evaluation_results)
    print(summary)

    with open("evaluation_summary.txt", 'w') as f:
        f.write(summary)

    print(
        f"Evaluation complete. Results saved to {RESULT_FILE}, evaluation_summary.csv, and evaluation_summary.txt")

if __name__ == "__main__":
    main()
