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


class VectorProcessor:
    """向量处理器：编码、白化和归一化"""
    
    def __init__(self, model: SentenceTransformer, is_whitening: bool = False, whitening_dim: int = 128, normalize: bool = False):
        self.model = model
        self.is_whitening = is_whitening
        self.whitening_dim = whitening_dim
        self.normalize = normalize
        self.whitening_model: Optional[VecsWhitening] = None
    
    def encode(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """文本编码为向量"""
        if not texts:
            raise ValueError("输入文本列表不能为空")
        
        vectors = self.model.encode(
            sentences=texts, 
            batch_size=64,
            device='cuda'
        )
        
        if self.is_whitening:
            vectors = self._apply_whitening(vectors)
        
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
        
        return self._to_float32(vectors)
    
    def _apply_whitening(self, vectors: np.ndarray) -> np.ndarray:
        """向量白化"""
        if self.whitening_model is None:
            self.whitening_model = VecsWhitening(n_components=self.whitening_dim)
            if vecs_whitening_pkl:
                self.whitening_model.load_bw_model(vecs_whitening_pkl)
                logger.info(f"加载白化模型，维度: {self.whitening_dim}")
            else:
                self.whitening_model.fit(vectors)
                logger.info(f"白化模型已初始化，维度: {self.whitening_dim}")
                self.whitening_model.save_bw_model(vecs_whitening_pkl)
        
        return self.whitening_model.transform(vectors)

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    @staticmethod
    def _to_float32(vecs):
        return vecs.astype(np.float32)


class IndexBuilder:
    """FAISS索引构建"""
    
    METRIC_MAP = {
        'cos': faiss.METRIC_INNER_PRODUCT,
        'l1': faiss.METRIC_L1,
        'l2': faiss.METRIC_L2,
        'l_inf': faiss.METRIC_Linf
    }
    
    @classmethod
    def build(cls, dimension: int, index_type: str = 'Flat', metric: str = 'cos') -> faiss.Index:
        metric_value = cls._get_metric_value(metric)
        
        if 'hnsw' in index_type.lower() and ',' not in index_type:
            hnsw_size = cls._extract_hnsw_size(index_type)
            index = faiss.IndexHNSWFlat(dimension, hnsw_size, metric_value)
        else:
            index = faiss.index_factory(dimension, index_type, metric_value)
        
        index.verbose = True
        index.do_polysemous_training = False
        
        return index
    
    @staticmethod
    def _get_metric_value(metric: str) -> int:
        if metric not in IndexBuilder.METRIC_MAP:
            supported = ', '.join(IndexBuilder.METRIC_MAP.keys())
            raise ValueError(f"不支持的度量方式: '{metric}'，支持: [{supported}]")
        return IndexBuilder.METRIC_MAP[metric]
    
    @staticmethod
    def _extract_hnsw_size(index_type: str) -> int:
        try:
            return int(index_type.lower().split('hnsw')[-1])
        except (ValueError, IndexError):
            raise ValueError(f"无法从 '{index_type}' 中提取HNSW参数")


class FaissSearcher:
    """FAISS检索系统"""
    def __init__(self, model_path: str, save_npy_path: str, docs: list, index_type: str = 'Flat',
                 measurement: str = 'cos', is_whitening: bool = False, whitening_dim: int = 128):
        self.docs = docs
        self.model_path = model_path
        self.save_npy_path = save_npy_path
        self.index_type = index_type
        self.measurement = measurement

        self.model = SentenceTransformer(self.model_path, device="cuda")
        normalize = (self.measurement == 'cos')
        self.vector_processor = VectorProcessor(
            model=self.model,
            is_whitening=is_whitening,
            whitening_dim=whitening_dim,
            normalize=normalize
        )
        
        self.index: Optional[faiss.Index] = None
        self.vecs: Optional[np.ndarray] = None
        self.sentences: List[str] = []
        self.vec_dim: Optional[int] = None
        
        self.train()
    
    def train(self):
        logger.info("向量化文档...")
        start_time = time.time()
        self.sentences = [doc.content for doc in self.docs]
        self.vecs = self.vector_processor.encode(self.sentences)
        self.vec_dim = self.vecs.shape[1]
        logger.info(f"向量化完成，耗时: {time.time() - start_time:.2f}秒")
        
        logger.info("构建索引...")
        self.index = IndexBuilder.build(
            dimension=self.vec_dim,
            index_type=self.index_type,
            metric=self.measurement
        )

        if not self.index.is_trained:
            self.index.train(self.vecs)

        self.index.add(self.vecs)
        logger.info(f"索引构建完成，数据量: {self.index.ntotal}")
        
    def search(self, queries: Union[str, List[str]], topK: int = 5) -> List[List[Dict[str, Any]]]:
        if isinstance(queries, str):
            queries = [queries]
        
        if not queries:
            return []
        
        query_vectors = self.vector_processor.encode(queries)
        distances, indices = self.index.search(query_vectors, topK)
        
        results = []
        for dist_row, idx_row in zip(distances, indices):
            query_results = []
            for score, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= len(self.docs):
                    continue
                doc = copy.deepcopy(self.docs[idx])
                doc.score = round(float(score), 4)
                query_results.append(doc.__dict__)
            results.append(query_results)
        
        return results
    
    def add_documents(self, new_docs: List):
        if not new_docs:
            return
        new_sentences = [doc.content for doc in new_docs]
        new_vectors = self.vector_processor.encode(new_sentences)
        
        self.index.add(new_vectors)
        
        nb_docs = len(self.docs)
        for i, _doc in enumerate(new_docs):
            _doc.id = nb_docs + i
        self.docs.extend(new_docs)
        self.sentences.extend(new_sentences)
        
        logger.info(f"已添加 {len(new_docs)} 个新文档到索引")


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