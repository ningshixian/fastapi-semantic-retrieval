import os
import sys
import re
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import gc
import atexit

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    InputExample
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import CoSENTLoss

# ================= 配置区域（需自行修改） ================= #
# 指定 GPU（例如使用第 1 张 GPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型路径配置（替换为你自己的本地路径或 HuggingFace Hub 地址）
pretrained_model_path = "path_or_hf_model_name"  # 预训练模型路径
model_cache_folder = "./model_cache"
save_ft_model_path = "./finetuned_models/sbert-finetuned"
train_similar_qq_path = "./data/train_similar_questions.csv"  # 训练数据路径

# ======================================================== #

# 打印 CUDA / PyTorch 信息
print("CUDA Available:", torch.cuda.is_available())
print("Torch Version:", torch.__version__)
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name())
    print("CUDA Version:", torch.version.cuda, " | GPU Count:", torch.cuda.device_count())
    print("Free GPU Memory:", torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    print("Current Device Index:", torch.cuda.current_device())
    print("NCCL Available:", torch.cuda.nccl.is_available(torch.rand(10).cuda()))

# 退出时清理显存
@atexit.register
def clean_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error cleaning memory: {e}")

# 数据加载与清理
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path, header=0, sep=",", encoding="utf-8", index_col=False)
        print("数据集 shape:", df.shape)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

    # 清理无效字符
    df = df.applymap(lambda x: re.sub(r"[\"”“]", "", str(x)))
    df = df.dropna()
    return df

# 负采样
def prepare_data(df, negative_sample_num=10):
    positive_df = pd.DataFrame()
    negative_df = pd.DataFrame()

    grouped = df.groupby("standard_question")
    for name, group in grouped:
        negative_samples = df[df["standard_question"] != name].sample(negative_sample_num)["similar_question"]
        negative_samples = pd.DataFrame({
            "standard_question": [name] * negative_sample_num,
            "similar_question": negative_samples
        })
        positive_df = pd.concat([positive_df, group])
        negative_df = pd.concat([negative_df, negative_samples])

    return positive_df, negative_df

def negative_sampling(df):
    positive_df, negative_df = prepare_data(df)
    positive_df['label'] = 1.0
    negative_df['label'] = 0.0
    all_examples_df = pd.concat([positive_df, negative_df])
    all_examples_df.rename(columns={
        "standard_question": "sentence1", 
        "similar_question": "sentence2"
    }, inplace=True)
    print("负采样后 shape:", all_examples_df.shape)

    from datasets import Dataset
    dataset = Dataset.from_dict({
        "sentence1": all_examples_df['sentence1'],
        "sentence2": all_examples_df['sentence2'],
        "label": all_examples_df['label'],
    })
    return dataset

# 1. 加载预训练模型
model = SentenceTransformer(
    pretrained_model_path,
    cache_folder=model_cache_folder,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 2. 加载数据并做负采样
df = load_and_clean_data(train_similar_qq_path)
dataset = negative_sampling(df)

# 3. 划分训练 & 测试集
dataset = dataset.train_test_split(test_size=0.1, seed=123, shuffle=True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# 4. 定义损失函数
cosent_loss = CoSENTLoss(model)

# 5. 定义训练参数
args = SentenceTransformerTrainingArguments(
    output_dir=save_ft_model_path,
    num_train_epochs=5,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_steps=500,
)

# 6. 训练器
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=cosent_loss,
)
trainer.train()

# 7. 保存最终模型
model.save_pretrained(save_ft_model_path + "/final")