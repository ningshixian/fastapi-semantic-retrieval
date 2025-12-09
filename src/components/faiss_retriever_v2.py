import os
import sys
import time
import copy
from typing import List, Union, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# 项目路径可按需添加
current_dir = Path(__file__).resolve().parent
parent_parent_dir = current_dir.parent
sys.path.append(str(parent_parent_dir))

# ===== 通用配置（示例，需用户自行替换为自己环境的配置路径/文件）=====
from configs.config import fine_tuned_model_config, vecs_whitening_pkl
from module.vecs_whitening import VecsWhitening
from utils import logger


"""
FAISS向量检索系统

该模块实现了基于FAISS的向量检索功能，支持多种索引类型和向量白化。
主要功能：
- 文本向量化和索引构建
- 向量白化处理
- 多种相似度度量方式
- 批量检索和实时检索
"""


class VectorProcessor:
    """向量处理器：编码、白化和归一化"""
    
    def __init__(
        self,
        model: SentenceTransformer,
        is_whitening: bool = False,
        whitening_dim: int = 128,
        normalize: bool = False
    ):
        self.model = model
        self.is_whitening = is_whitening
        self.whitening_dim = whitening_dim
        self.normalize = normalize
        self.whitening_model: Optional[VecsWhitening] = None
    
    def encode(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """将文本编码为向量"""
        if not texts:
            raise ValueError("输入文本列表不能为空")
        
        # 编码文本 - 使用批量处理提高性能
        vectors = self.model.encode(
            sentences=texts, 
            batch_size=64, 
            # show_progress_bar=True, 
            device='cuda'
        )
        
        # 白化处理 (当数据量大的时候，非常耗时！)
        if self.is_whitening:
            vectors = self._apply_whitening(vectors)
        
        # 归一化处理 (当数据量大的时候，非常耗时！)
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
        
        return self._to_float32(vectors)
    
    def _apply_whitening(self, vectors: np.ndarray) -> np.ndarray:
        """简单的向量白化改善句向量质量 
        https://github.com/bojone/BERT-whitening"""
        if self.whitening_model is None:
            self.whitening_model = VecsWhitening(n_components=self.whitening_dim)
            # 加载白化模型
            if vecs_whitening_pkl:
                self.whitening_model.load_bw_model(vecs_whitening_pkl)
                logger.info(f"本地加载白化模型pkl，维度: {self.whitening_dim}")
            else:
                self.whitening_model.fit(vectors)
                logger.info(f"白化模型已初始化，维度: {self.whitening_dim}")
                self.whitening_model.save_bw_model(vecs_whitening_pkl)
        
        return self.whitening_model.transform(vectors)

    # @staticmethod
    # def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    #     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    #     norms = np.where(norms == 0, 1, norms)
    #     return vectors / norms

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """向量归一化
        性能优化 → 使用numpy内置函数"""
        # 使用numpy的linalg.norm进行更高效的归一化
        # 对于大量向量，这种方法比手动计算更高效
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除以零的情况
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    @staticmethod
    def _to_float32(vecs):
        """将向量转换为float32类型"""
        return vecs.astype(np.float32)


class IndexBuilder:
    """FAISS索引构建"""
    
    # 支持的度量方式映射
    METRIC_MAP = {
        'cos': faiss.METRIC_INNER_PRODUCT,
        'l1': faiss.METRIC_L1,
        'l2': faiss.METRIC_L2,
        'l_inf': faiss.METRIC_Linf,
        'brayCurtis': faiss.METRIC_BrayCurtis,
        'canberra': faiss.METRIC_Canberra,
        'jensen_shannon': faiss.METRIC_JensenShannon
    }
    
    @classmethod
    def build(
        cls,
        dimension: int,
        index_type: str = 'Flat',
        metric: str = 'cos'
    ) -> faiss.Index:
        """构建FAISS索引"""
        metric_value = cls._get_metric_value(metric)
        
        if 'hnsw' in index_type.lower() and ',' not in index_type:
            # HNSW索引
            hnsw_size = cls._extract_hnsw_size(index_type)
            index = faiss.IndexHNSWFlat(dimension, hnsw_size, metric_value)
        else:
            # 其他索引类型
            index = faiss.index_factory(dimension, index_type, metric_value)
        
        index.verbose = True
        index.do_polysemous_training = False
        
        return index
    
    @staticmethod
    def _get_metric_value(metric: str) -> int:
        """获取度量方式对应的FAISS常量"""
        if metric not in IndexBuilder.METRIC_MAP:
            supported = ', '.join(IndexBuilder.METRIC_MAP.keys())
            raise ValueError(f"不支持的度量方式: '{metric}'，支持: [{supported}]")
        return IndexBuilder.METRIC_MAP[metric]
    
    @staticmethod
    def _extract_hnsw_size(index_type: str) -> int:
        """从索引类型字符串中提取HNSW大小参数"""
        try:
            return int(index_type.lower().split('hnsw')[-1])
        except (ValueError, IndexError):
            raise ValueError(f"无法从 '{index_type}' 中提取HNSW参数")


class FaissSearcher:
    """FAISS检索系统
    支持多种索引类型和相似度度量方式，可选向量白化处理。
    """
    def __init__(
            self, 
            model_path: str, 
            save_npy_path: str,
            docs: list, 
            index_type: str = 'Flat',  # 'HNSW64' 基于图检索，检索速度极快，且召回率几乎可以媲美Flat
            measurement: str = 'cos',
            is_whitening: bool = False,
            whitening_dim: int = 128,  # 为啥越小越好？
            # norm_vec: bool = False,
            **kwargs
    ):
        # 初始化方法
        # self.qa_path_list = qa_path_list
        self.docs = docs
        self.model_path = model_path
        self.save_npy_path = save_npy_path
        self.index_type = index_type
        self.measurement = measurement

        # 初始化句子嵌入模型
        # import torch
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_path, device="cuda")
        # self.measurement = self._set_measure_metric(measurement)

        # 初始化向量处理器
        normalize = (self.measurement == 'cos')  # 余弦相似度需要归一化
        self.vector_processor = VectorProcessor(
            model=self.model,
            is_whitening=is_whitening,
            whitening_dim=whitening_dim,
            normalize=normalize
        )

        # 索引相关属性
        self.index: Optional[faiss.Index] = None
        self.vecs: Optional[np.ndarray] = None
        self.sentences: List[str] = []
        self.vec_dim: None
        
        # 构建faiss索引
        self.train()
    
    def train(self):
        """训练索引流程"""
        logger.info(f"1、加载qa数据集并进行向量化...")
        start_time = time.time()
        # 提取句子内容
        self.sentences = [doc.content for doc in self.docs]
        # 向量化
        self.vecs = self.vector_processor.encode(self.sentences)
        self.vec_dim = self.vecs.shape[1]
        logger.info(f"向量化完成，耗时: {time.time() - start_time:.2f}秒，句子数量: {len(self.sentences)}")

        logger.info("2、构建索引...")
        start_time = time.time()
        self.index = IndexBuilder.build(
            dimension=self.vec_dim,
            index_type=self.index_type,
            metric=self.measurement
        )

        # 训练索引（如果需要）
        if not self.index.is_trained:    # 输出为True，代表该类index不需要训练，只需要add向量进去即可
            self.index.train(self.vecs)

        # 添加向量到索引
        self.index.add(self.vecs)

        elapsed_time = time.time() - start_time
        logger.info(
            f"索引构建完成！耗时: {elapsed_time:.2f}秒，"
            f"索引数据量: {self.index.ntotal}"
        )
        
    def search(
        self,
        queries: Union[str, List[str]],
        topK: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        执行向量检索
        
        Args:
            queries: 查询文本或查询文本列表
            topK: 返回最相似的K个结果
            
        Returns:
            检索结果列表，每个查询对应一个结果列表
        """
        # 参数标准化
        if isinstance(queries, str):
            queries = [queries]
        
        if not queries:
            return []
        
        # 编码查询向量
        query_vectors = self.vector_processor.encode(queries)
        
        # 执行检索
        distances, indices = self.index.search(query_vectors, topK)
        
        # 处理向量检索结果
        # 将FAISS返回的距离和索引转换为实际的文档结果
        results = []
        for query, dist_row, idx_row in zip(queries, distances, indices):
            query_results = []
            
            for score, idx in zip(dist_row, idx_row):
                # 检查索引是否有效
                if idx < 0 or idx >= len(self.docs):
                    # logger.warning(f"索引 {idx} 超出文档范围 [0, {len(self.docs)})")
                    continue
                
                # 复制文档对象并添加检索信息
                doc = copy.deepcopy(self.docs[idx])
                doc.score = round(float(score), 4)
                if doc.id != idx:   # 不生效（不会筛掉doc，warning信息也不会记录）
                    logger.warning(f"文档ID {doc.id} 与索引 {idx} 不匹配")
                query_results.append(doc.__dict__)    # document 对象转成字典
            
            results.append(query_results)
        
        return results
    
    def add_documents(self, new_docs: List):
        """向索引中添加新文档
        优化点：
        1. 解决 O(N*M) 性能问题，使用哈希表实现 O(N+M)。
        2. 支持处理 self.docs 中存在多个相同 question_id 的情况（全部标记删除）。
        3. 支持处理 new_docs 本次批次内部的重复 question_id（保留最新的，旧的标记删除）。
        4. 增加事务安全性，防止索引与数据不一致。
        """
        if not new_docs:
            return

        # 1. 预处理：提取内容并编码 (Fail Fast: 如果编码失败，不污染索引)
        try:
            new_sentences = [doc.content for doc in new_docs]
            new_vectors = self.vector_processor.encode(new_sentences)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return
        
        # 2. 构建现有文档的“一对多”查找表
        existing_qid_map = defaultdict(list)
        target_qids = {d.meta.get('question_id') for d in new_docs} - {None}
        if target_qids:
            for idx, doc in enumerate(self.docs):
                q_id = doc.meta.get('question_id')
                # 核心逻辑：ID命中 且 未被标记删除 (is_delete != 1)
                if q_id in target_qids and doc.meta.get('is_delete') != 1:
                    existing_qid_map[q_id].append(idx)

        # 3. 处理新文档逻辑
        current_doc_len = len(self.docs)

        for i, new_doc in enumerate(new_docs):
            new_doc.id = current_doc_len + i
            q_id = new_doc.meta.get('question_id')
            # 12.1更新：检查是否存在旧知识 (O(1) 查找)
            # 存在（修改/删除/禁用），则将旧知识状态 is_delete 改为 1
            if q_id in existing_qid_map:
                for old_index in existing_qid_map[q_id]:
                    self.docs[old_index].meta['is_delete'] = 1  # 标记旧文档为删除
                # 标记完后，从 map 中移除，避免 new_docs 里有重复 ID 时重复遍历
                del existing_qid_map[q_id]
        
        # 4. 更新内存数据
        self.docs.extend(new_docs)
        self.sentences.extend(new_sentences)
        
        # 5. 添加到向量索引
        try:
            self.index.add(new_vectors)
            logger.info(f"已添加 {len(new_docs)} 个新文档到索引")
        except Exception as e:
            # 回滚操作：如果索引添加失败，移除刚刚添加到内存列表的数据
            self.docs = self.docs[:-len(new_docs)]
            self.sentences = self.sentences[:-len(new_sentences)]
            logger.error(f"Failed to add to index, rolled back: {e}")
            raise e


if __name__ == "__main__":
    @dataclass
    class Document:
        id: str = field(default="")
        content: Optional[str] = field(default=None)
        meta: Dict[str, Any] = field(default_factory=dict)
        score: Optional[float] = field(default=None)

    # 伪造示例文档（无业务隐私）
    documents = [
        Document(id=0, content="文档一内容", meta={"tag": "demo"}),
        Document(id=1, content="文档二内容", meta={"tag": "demo"}),
        Document(id=2, content="文档三内容", meta={"tag": "demo"})
    ]
    new_docs = [
        Document(id=3, content="新增文档A", meta={"tag": "test"}),
        Document(id=4, content="新增文档B", meta={"tag": "test"})
    ]
    
    # 使用通用配置（用户需替换为自己可用的模型路径）
    model_type = "your_model_key"
    save_ft_model_path, npy_path = fine_tuned_model_config[model_type]
    
    searcher = FaissSearcher(
        model_path=save_ft_model_path,
        save_npy_path=npy_path,
        docs=documents,
        index_type="Flat",
        measurement="cos",
        is_whitening=True,
        whitening_dim=128
    )
    
    print("检索结果:", searcher.search("测试查询", topK=2))
    searcher.add_documents(new_docs)
    print("添加新文档后:", searcher.search("测试查询", topK=2))