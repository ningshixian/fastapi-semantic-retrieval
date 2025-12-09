import os
import sys
import copy
from collections import defaultdict
from typing import List, Union
from numpy import ndarray
import torch
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# 静默导入，避免多余输出
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    from rerankers import Reranker
    from rerankers import Document

# 获取当前文件的绝对路径及上一级路径
current_dir = Path(__file__).resolve().parent
parent_parent_dir = current_dir.parent
sys.path.append(str(parent_parent_dir))

# 模型路径（需自行配置）
from configs.config import rerank_model_path  # 请在配置文件中定义

"""
通用的多路召回结果重排与融合类
参考: https://github.com/AnswerDotAI/rerankers
"""

class PostRank:
    def __init__(self):
        # 初始化交叉编码器（Cross-Encoder）
        self.ranker = Reranker(
            rerank_model_path,
            model_type='cross-encoder', 
            device="cuda",
            # 可选参数:
            # verbose=0, 
            # dtype=torch.float16, 
            # batch_size=64, 
            # lang="xx",    # 语言代码
        )

    def rerank(
        self,
        queries: Union[List[str], ndarray],
        merge_hits: List,
        semantic_hits: List,
        fusion: bool = True,
    ):
        """对多路召回结果进行重排和加权融合"""
        semantic_hits_copy = copy.deepcopy(semantic_hits)  # 避免修改原数据
        merge_hits_copy = copy.deepcopy(merge_hits)

        # 对每个 query 进行独立重排
        results = [
            self.rerank4query(query, docs)
            for query, docs in zip(queries, merge_hits_copy)
        ]
        
        # 可选：进行加权融合
        if fusion and semantic_hits:
            results = [
                self.weighted_fusion(semantic_hit, rank_hit, weights=[0.5, 0.5])
                for semantic_hit, rank_hit in zip(semantic_hits_copy, results)
            ]
        
        return results

    def rerank4query(self, query, docs):
        """
        使用 reranker 对召回的文档结果进行重排。
        参数:
            docs: 文档列表，每个文档是一个字典，至少包含 'content' 字段。
            query: 查询字符串。
        返回:
            按分数降序排序的文档列表。
        """
        if not isinstance(docs, list) or not isinstance(query, str):
            raise ValueError("docs 必须是列表，query 必须是字符串。")
        
        # 若候选结果只有一个，直接返回，防止ranker.rank报错 (https://github.com/AnswerDotAI/rerankers/pull/9)
        # 注意，需保证BM25召回路的分值已提前归一化
        if len(docs) == 1:
            return docs

        # 构建 cross-encoder 输入
        _docs = [
            Document(text=doc["content"], doc_id=did)
            for did, doc in enumerate(docs)
        ]
        results = self.ranker.rank(query=query, docs=_docs)

        # 分数归一化（Sigmoid 输出）
        score_logits = [result.score for result in results]
        scores = torch.sigmoid(torch.tensor(score_logits)).tolist()  # 范围 0~1
        
        # 得分归一化 - sigmoid
        new_docs = []
        score_logits = [result.score for result in results]
        scores = torch.sigmoid(torch.tensor(score_logits)).tolist()  # 0~1 
        for ix, res in enumerate(results):
            docs[res.doc_id]["score"] = scores[ix]
            new_docs.append(docs[res.doc_id])

        new_docs.sort(key=lambda x: x["score"], reverse=True)
        return new_docs

    def weighted_fusion(self, hits1, hits2, weights=[0.5, 0.5]):
        """
        对两路召回结果进行加权融合
        """
        combined_results = defaultdict(lambda: {"score": 0})
        
        # 第一组召回结果加权
        for hit in hits1:
            hit["score"] *= weights[0]
            item_id = hit["meta"]["id"]  # 泛化后的唯一标识
            combined_results[item_id] = hit

        # 第二组召回结果加权并合并
        for hit2 in hits2:
            hit2['score'] *= weights[1]
            item_id2 = hit2["meta"]["id"]
            if item_id2 in combined_results:
                combined_results[item_id2]['score'] += hit2['score']
            else:
                combined_results[item_id2] = hit2

        # 按分数降序排序并返回结果
        combined_results = sorted(
            combined_results.items(), key=lambda x: x[1]["score"], reverse=True
        )
        return [x[1] for x in combined_results]