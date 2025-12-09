import os
import copy
from typing import Union, List
from collections import OrderedDict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import bm25s
from bm25s.tokenization import Tokenized
# from rank_bm25 import BM25Okapi

import logging
import jieba
jieba.setLogLevel(logging.INFO)


def tokenize(
    texts,
    return_ids: bool = True,
    show_progress: bool = False,
    leave: bool = False,
) -> Union[List[List[str]], Tokenized]:
    if isinstance(texts, str):
        texts = [texts]

    corpus_ids = []
    token_to_index = {}

    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):

        splitted = jieba.lcut(text, HMM=False)
        doc_ids = []

        for token in splitted:
            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)

            token_id = token_to_index[token]
            doc_ids.append(token_id)

        corpus_ids.append(doc_ids)

    # Create a list of unique tokens that we will use to create the vocabulary
    unique_tokens = list(token_to_index.keys())

    vocab_dict = token_to_index

    # Return the tokenized IDs and the vocab dictionary or the tokenized strings
    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=vocab_dict)

    else:
        # We need a reverse dictionary to convert the token IDs back to tokens
        reverse_dict = unique_tokens
        # We convert the token IDs back to tokens in-place
        for i, token_ids in enumerate(
            tqdm(
                corpus_ids,
                desc="Reconstructing token strings",
                leave=leave,
                disable=not show_progress,
            )
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids


bm25s.tokenize = tokenize   # 暂未使用
# bm25s.tokenize(self.sentences)  # , stopwords="zh", show_progress=False


# Min-Max 归一化，适合与余弦相似度进行加权求和
def normalize_min_max(scores):
    min_val = np.min(scores, keepdims=True)
    max_val = np.max(scores, keepdims=True)
    diff = max_val - min_val

    # 核心逻辑：
    # 1. 如果 diff 为 0，返回 1.0
    # 2. 如果 diff 不为 0，计算 (score - low) / diff
    # 注意：分母加 1e-9 只是为了防止 Python 报 "除以0" 的警告，
    # 实际上 diff==0 时，np.where 会直接取 1.0，不会用到除法的结果。
    return np.where(diff == 0, 1.0, (scores - min_val) / (diff + 1e-9))


# Softmax 归一化
def normalize_softmax(scores, temperature=1.0):
    # 为了数值稳定性，先减去最大值
    exp_scores = np.exp((scores - np.max(scores, keepdims=True)) / temperature)
    return exp_scores / exp_scores.sum(keepdims=True)


# # 测试
# scores = [1.5, 2.5, 15]
# print(normalize_min_max(scores))
# # 使用较高的温度系数使分布更平滑
# print(normalize_softmax(scores, temperature=5.0))
# exit()



class BM25Model:
    def __init__(self, docs: list):
        self.docs = docs
        self.sentences = []
        self.tokens = []
        # 初始化BM25
        self.retriever = None
        self.load_retriever()
    
    def _tokenize_text(self, texts: list[str]) -> list[list[str]]:
        """
        自定义分词函数
        bm25s.tokenize 返回的是对象，不方便合并，所以我们自己分词
        """
        return [jieba.lcut(text) for text in texts]

    def load_retriever(self):
        self.sentences = [doc.content for doc in self.docs]
        # 1. 分词 (耗时操作)
        self.tokens = self._tokenize_text(self.sentences)
        # 2. 建立索引 (快速操作)
        self.retriever = bm25s.BM25(
            method="bm25+", delta=1.5, 
            corpus=self.sentences   # 可省略这个参数
        )
        self.retriever.index(self.tokens)
        # logger.info("BM25: 初始化完成")
    
    def add_documents(self, new_docs: list):
        """
        增量更新函数
        策略：只对新文档分词 -> 合并 Token 列表 -> 快速重索引
        """
        if not new_docs:
            return
        
        # 1. 仅对【新增】文档进行分词 (节省大量时间)
        new_texts = [doc.content for doc in new_docs]
        new_tokens = self._tokenize_text(new_texts)
        # 2. 更新内存中的主数据
        self.sentences.extend(new_texts)
        self.tokens.extend(new_tokens)
        # 3. 重新索引
        # bm25s 的 index 速度极快 (10万条数据通常在 1秒内)，因此全量重索引是可以接受的
        self.retriever.index(self.tokens)
        
        # logger.info(f"BM25: 增量更新完成，当前总文档数: {len(self.all_docs)}")

    def bm25_similarity(self, queries:Union[str, List[str]], topK=10):
        if not self.sentences:
            raise ValueError("corpus is None. Please add_corpus first, eg. `add_corpus(corpus)`")
        if not self.retriever:
            self.load_retriever()
        if isinstance(queries, str):
            queries = [queries]
        
        topK = min(topK, len(self.sentences))  # raise ValueError when the corpus size is less than k. fix #117
        query_list = {id: query for id, query in enumerate(queries)}
        result = {query_id: {} for query_id, query in query_list.items()}

        query_tokens = self._tokenize_text(queries)  # 分词
        idx_range = list(range(len(self.sentences)))    # 索引范围
        _documents, _scores = self.retriever.retrieve(
            query_tokens, 
            corpus=idx_range,   # 输出索引
            k=topK, 
            show_progress=False
        )
        # 对_scores进行归一化
        _scores = [normalize_min_max(_score) for _score in _scores]

        # 处理检索结果
        closest_matches = []
        for i in range(len(queries)):
            matches = []
            for j in range(topK):
                idx = _documents[i][j]
                score = round(float(_scores[i][j]), 4)
                doc = copy.deepcopy(self.docs[idx])
                doc.score = score
                # assert doc.id == idx
                matches.append(doc.__dict__)
            closest_matches.append(matches)

        return closest_matches


# if __name__ == '__main__':

#     from configs.config import *
#     BM25 = BM25Model(qa_path_list=[qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting],)
#     query = ["天窗可以打开吗", "明天天气怎么样"]
#     print(BM25.bm25_similarity(query, topK=3))


#     # Happy with your index? Save it for later...
#     # retriever.save("bm25s_index_animals")

#     # # ...and load it when needed
#     # ret_loaded = bm25s.BM25.load("bm25s_index_animals", load_corpus=True)
