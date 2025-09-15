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

bm25s.tokenize = tokenize


def min_max_normalization(scores):
    min_value = np.min(scores)
    max_value = np.max(scores)
    if min_value == max_value:
        # raise ValueError("Minimum and maximum values are the same, cannot normalize.")
        return scores
    # 应用最小-最大标准化公式
    normalized_data = (scores - min_value) / (max_value - min_value)
    return normalized_data


def normalization10(scores):
    # 将 BM25 分数除以本身加上 10
    normalized_data = scores / (scores + 10)
    return normalized_data


class BM25Model:
    def __init__(self, docs: list):
        self.docs = docs
        self.qid_dict = {}
        self.sen2qid = OrderedDict()    # 其实无序字典亦可
        self.sentences = []
        self.retriever = None
        self.load_retriever()

    def load_retriever(self):
        self.sentences = [doc.content for doc in self.docs]
        # Tokenize the corpus and index it
        corpus_tokens = bm25s.tokenize(self.sentences)
        self.retriever = bm25s.BM25(corpus=self.sentences, method="bm25+")
        self.retriever.index(corpus_tokens)

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

        query_tokens = bm25s.tokenize(queries)
        idx_range = list(range(len(self.sentences)))    # 索引范围
        _documents, _scores = self.retriever.retrieve(
            query_tokens, 
            corpus=idx_range,   # 输出索引
            k=topK, 
            show_progress=False
        )

        # 处理检索结果
        closest_matches = []
        for i in range(len(queries)):
            matches = []
            for j in range(topK):
                idx = _documents[i][j]
                score = round(float(_scores[i][j]), 4)
                doc = copy.deepcopy(self.docs[idx])
                doc.score = score
                assert doc.id == idx
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
