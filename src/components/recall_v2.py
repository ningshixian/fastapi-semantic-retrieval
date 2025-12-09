import sys
import json
import traceback
from collections import defaultdict
from math import inf
from pathlib import Path
from typing import Dict, List

# ===== 路径设置（通用化） =====
# 获取当前文件的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取上一级目录
parent_dir = current_dir.parent
# 将上一级目录添加到 sys.path
sys.path.append(str(parent_dir))

# ===== 模型配置（业务部分已去掉） =====
# 示例：fine_tuned_model_config["embedding_model"] = ("path/to/model", "path/to/vecs.npy")
from configs.config import fine_tuned_model_config  

# ===== 引入通用检索模块（保留结构） =====
from module.faiss_retriever_v2 import FaissSearcher
from module.bm25.bm25_sparse_v2 import BM25Model


class Recall:
    def __init__(self, docs, model_type="embedding_model", is_whitening=True, whitening_dim=128):
        """
        docs: 文档列表
        model_type: 模型类型标识，用于查找配置
        """
        self.is_whitening = is_whitening
        self.docs = docs
        
        # 读取模型路径
        model_path, npy_path = fine_tuned_model_config[model_type]

        # 初始化 Faiss 搜索器
        self.faiss = FaissSearcher(
            model_path=model_path,
            save_npy_path=npy_path,
            docs=self.docs,
            index_param='Flat',    # 精确检索，可选 HNSW
            measurement='cos',     # 余弦相似度
            is_whitening=is_whitening,
            whitening_dim=whitening_dim,
        )

        # 初始化 BM25 搜索
        self.bm25 = BM25Model(docs=self.docs)
    
    def retrieve(self, text, topK, search_strategy):
        """
        text: 查询文本（str 或 [str]）
        topK: 检索数量
        search_strategy: 'hybrid'、'bm25'、'embedding'
        """
        if isinstance(text, str):
            text = [text]

        if search_strategy == 'hybrid':
            # 双路召回
            bm25_hits = self.bm25.bm25_similarity(text, topK=topK * 5)
            semantic_hits = self.faiss.search(text, topK=topK * 5)
            merge_hits = [remove_duplicates(a + b) for a, b in zip(bm25_hits, semantic_hits)]
            semantic_hits = [remove_duplicates(a) for a in semantic_hits]
            return merge_hits, semantic_hits

        elif search_strategy == 'bm25':
            bm25_hits = self.bm25.bm25_similarity(text, topK=topK * 10)
            bm25_hits = [remove_duplicates(a) for a in bm25_hits]
            return bm25_hits

        elif search_strategy == 'embedding':
            semantic_hits = self.faiss.search(text, topK=topK * 10)
            semantic_hits = [remove_duplicates(a) for a in semantic_hits]
            return semantic_hits


def remove_duplicates(docs: List[Dict]):
    """去重：同 question_id 保留得分最高的文档，分数相同时保留 id 较大的"""
    docs_by_id = defaultdict(list)
    for doc in docs:
        idx = doc["id"]
        question_id = doc["meta"]["question_id"]
        docs_by_id[question_id].append((idx, doc))
    # 选择最优文档：分数最高，同分时id索引最大
    result = []
    for group in docs_by_id.values():
        _, best_doc = max(group, key=lambda x: (x[1].get("score", -inf), x[0]))
        result.append(best_doc)

    # 按分数降序排序
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


# ===== 测试去重功能（示例） =====
if __name__ == "__main__":
    test_docs = [
        {"id": 0, "content": "文档1内容", "score": 0.85, "meta": {"question_id": "Q001"}},
        {"id": 1, "content": "文档2内容", "score": 0.90, "meta": {"question_id": "Q002"}},
        {"id": 2, "content": "文档3内容 - Q001重复", "score": 0.95, "meta": {"question_id": "Q001"}},
        {"id": 3, "content": "文档2不同版本", "score": 0.90, "meta": {"question_id": "Q002"}},
    ]

    result = remove_duplicates(test_docs)
    print(f"原始文档数: {len(test_docs)}")
    print(f"去重后文档数: {len(result)}")
    for r in result:
        print(r)