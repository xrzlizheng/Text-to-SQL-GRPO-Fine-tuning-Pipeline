# Text-to-SQL GRPO 微调流程
欢迎关注我的博客https://blog.csdn.net/qq_36603091/article/details/147732147
本仓库包含使用通用奖励近端优化（GRPO）对大型语言模型（LLMs）进行 Text-to-SQL 转换微调的流程。该实现主要针对 Qwen2.5-Coder 模型，但也可适用于其他 LLMs。

## 概述

Text-to-SQL 是将自然语言问题转换为 SQL 查询的任务。本项目使用 GRPO 进行模型微调，优化以下方面：
- SQL 正确性
- 清晰的推理过程
- 适当的格式
- 查询复杂度匹配

## 主要特点

- **GRPO 微调**：使用多个奖励函数优化模型
- **评估**：使用标准查询和 GPT-4o-mini 的综合评估框架
- **SQL 奖励函数**：用于 SQL 质量评估的多个奖励指标
- **对比学习**：改进 SQL 生成的自然语言理解能力

## 项目结构

- `llm_train.py`：GRPO 微调的主要训练脚本
- `sql_reward_utils.py`：SQL 执行和奖励函数
- `eval_grpo.py`：微调模型的评估
- `requirements.txt`：所需依赖项

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

1. 清理数据集：
```bash
python cleanse_dataset.py
```

该脚本过滤数据集以确保：
- 有效的 SQL 查询
- 正确匹配的数据库模式上下文
- 具有正确语法的可执行查询

## 训练

运行 GRPO 训练：

```bash
python llm_train.py
```

关键参数（可在脚本中修改）：
- `MODEL_NAME`：要微调的基础模型（默认："Qwen/Qwen2.5-Coder-7B-Instruct"）
- `MAX_SEQ_LENGTH`：最大序列长度（默认：1024）
- `LORA_RANK`：参数高效微调的 LoRA 秩（默认：32）
- `BATCH_SIZE`：训练批次大小（默认：4）
- `NUM_GENERATIONS`：每个提示的 GRPO 生成数量（默认：8）
- `MAX_STEPS`：最大训练步数（默认：225）

## 评估

评估您的训练模型：

```bash
python eval_grpo.py
```

该脚本：
1. 加载您的微调模型
2. 从测试提示生成 SQL 查询
3. 使用 GPT-4o-mini 评估输出
4. 生成详细的指标和错误分析
5. 将结果保存为 JSON 和 CSV 格式

## 奖励函数

训练使用多个奖励组件：

- **格式奖励**：确保正确的 XML 标签结构
- **SQL 正确性**：与标准查询比较可执行准确性
- **复杂度奖励**：匹配生成查询和标准查询之间的复杂度
- **推理质量**：评估解释质量和数据库模式引用

## 模型输出

模型被训练为以下格式输出：

```
<reasoning>
这个数据库有一个包含 id、name 和 age 列的 users 表。
问题要求查询所有 30 岁以上的用户，所以我需要使用 WHERE 条件查询 users 表。
</reasoning>
<sql>
SELECT * FROM users WHERE age > 30;
</sql>
```
