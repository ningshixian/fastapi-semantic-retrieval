import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import logging
import time
import datetime
import re
import string
import json
import traceback
from zhon.hanzi import punctuation

# 导入本地库（视情况调整路径）
import sys
sys.path.append(r"../")
from utils import clean_text
from configs.config import paths, engine  # engine 为 SQLAlchemy 数据库连接引擎

print("==================获取【问题数据】==================")
with open("src/connections/db/sql/question.sql", "r") as f:
    sql_question = f.read()

with engine.connect() as conn:
    ret = pd.read_sql(sql_question, conn)
print("question数量: ", ret.shape)

print("==================获取【答案数据】==================")
with open("src/connections/db/sql/answer.sql", "r") as f:
    sql_answer = f.read()

# 答案类型（可根据实际业务调整）
answer_types = ('type_a', 'type_b', 'type_c')
placeholders = ','.join(['%s'] * len(answer_types))
sql_answer = sql_answer.replace('IN (%s)', f'IN ({placeholders})')

_start_time = time.time()
with engine.connect() as conn:
    ret2 = pd.read_sql(sql_answer, conn, params=answer_types)
print('--耗时: %s' % (time.time() - _start_time))
print("answer数量: ", ret2.shape)

print("==================获取【相似问数据】==================")
with open("src/connections/db/sql/similar_question.sql", "r") as f:
    sql_query3 = f.read()

with engine.connect() as conn:
    ret3 = pd.read_sql(sql_query3, conn)
print("相似问数量: ", ret3.shape)

print("==================数据清洗与合并==================")
def clean_dataframe(df, text_column=None, time_columns=None):
    """
    通用的数据清洗函数
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    
    if text_column:
        df[text_column] = df[text_column].apply(clean_text)
    
    if time_columns:
        df[time_columns] = df[time_columns].astype(str).replace('NaT', '')
    
    df = df.fillna("")
    return df.reset_index(drop=True)

def filter_questions(df, col):
    """
    过滤不符合条件的问题，例如纯符号、纯字母数字等
    """
    bracket_pattern = r'^\[.*\]$'
    alphanumeric_pattern = rf'^[a-zA-Z0-9\s{string.punctuation}{punctuation}]*$'
    
    filters = (
        (df[col] != '') &
        (df[col].str.len() > 2) &
        (~df[col].str.contains("示例屏蔽词1|示例屏蔽词2", na=False, regex=True)) &
        (~df[col].str.match(bracket_pattern, na=False)) &
        (~df[col].str.match(alphanumeric_pattern, na=False))
    )
    return df[filters].reset_index(drop=True)

def merge_data(question_df, answer_df, sim_df):
    try:
        # 处理问题数据
        question_df = clean_dataframe(
            question_df, 
            text_column='question_content',
            time_columns=['valid_begin_time', 'valid_end_time']
        )
        question_df = filter_questions(question_df, 'question_content')

        # 处理答案数据
        answer_df = clean_dataframe(answer_df, time_columns=['valid_begin_time', 'valid_end_time'])
        fill_columns = ['answer_type_list', 'label_list']
        answer_df[fill_columns] = answer_df[fill_columns].applymap(
            lambda x: json.loads(x) if pd.notna(x) and x else []
        )

        # 处理相似问数据
        sim_df = clean_dataframe(sim_df, text_column='similar_question')

        # 构建相似问字典
        sim_dict = sim_df.groupby('question_id').apply(lambda x: x.to_dict('records')).to_dict()

        # 构建答案字典
        from collections import defaultdict
        type_a_answers = defaultdict(list)
        type_b_answers = defaultdict(list)
        for qid, group in answer_df.groupby('question_id'):
            for _, row in group.iterrows():
                _answer_dict = {
                    "answer_id": row["answer_id"],
                    "answer_content": row["answer_content"],
                    "answer_type_list": row["answer_type_list"],
                    "label_list": row["label_list"],
                    "status": row["answer_status"],
                    "valid_begin_time": row["valid_begin_time"],
                    "valid_end_time": row["valid_end_time"],
                    "is_default": row["is_default_answer"],
                }
                answer_types_joined = ",".join(row["answer_type_list"])
                if 'type_c' in answer_types_joined:
                    type_a_answers[qid].append(_answer_dict)
                if 'type_a' in answer_types_joined or 'type_b' in answer_types_joined:
                    type_b_answers[qid].append(_answer_dict)

        # 构建最终 JSON 对象
        a_merged_data = []
        b_merged_data = []
        for _, row in question_df.iterrows():
            qid = row["question_id"]
            if qid not in {**type_a_answers, **type_b_answers}:
                continue
            
            _data = {
                "answer_content_list": [],
                "id": qid,
                "question_id": qid,
                "question_content": row["question_content"],
                "question_type": row["question_type"],
                "similar_question_list": sim_dict.get(qid, []),
                "category_all_name": row["category_all_name"],
                "status": 1,
                "valid_begin_time": row["valid_begin_time"],
                "valid_end_time": row["valid_end_time"],
                "source": "某知识平台",
            }

            if qid in type_a_answers:
                _data["answer_content_list"] = type_a_answers[qid]
                a_merged_data.append(_data)
            if qid in type_b_answers:
                _data["answer_content_list"] = type_b_answers[qid]
                b_merged_data.append(_data)

        with open(paths.qa.type_a, "w", encoding="utf-8") as f:
            json.dump(a_merged_data, f, ensure_ascii=False, indent=4)
            print(f"{len(a_merged_data)} 条 type_a 数据写入文件")
        with open(paths.qa.type_b, "w", encoding="utf-8") as f:
            json.dump(b_merged_data, f, ensure_ascii=False, indent=4)
            print(f"{len(b_merged_data)} 条 type_b 数据写入文件")
        
    except Exception:
        traceback.print_exc()

# 执行数据合并
merge_data(ret, ret2, ret3)