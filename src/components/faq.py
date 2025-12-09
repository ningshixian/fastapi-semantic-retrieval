import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# 通用召回与排序模块（请在项目中自行实现/引入）
from recall_module import Recall
from rank_module import PostRank
from utils import logger
from configs.config import rerank_model_path

"""
提供 FAQ 基础检索服务，包括：数据转换、检索召回、排序
"""

# 全局 Ranker 实例（所有 FAQ 实例共享）
GLOBAL_RANK_MODULE = PostRank()
logger.info(f"PostRank model path: {rerank_model_path}")


@dataclass
class Document:
    """基础数据类——用作检索文档容器"""
    id: str = field(default="")
    content: Optional[str] = field(default=None)
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = field(default=None)


class DataPreprocessor:
    """数据预处理模块：将原始 QA 结构转换为 Document 列表"""
    def _convert_qa_to_document(self, qa_item: Dict[str, Any]) -> List[Document]:
        documents = []
        main_question = qa_item.get("question_content", "").strip()
        
        # 问题元信息（自行根据业务场景扩展）
        question_meta = {
            "question_id": qa_item.get("question_id"),              # 问题ID
            "question_type": qa_item.get("question_type", 0),          # 问题类型（0标准问题、1知识点）
            "main_question": main_question,                         # 标注问（standard_sentence）
            # "long_effective": qa_item.get("longEffective", 0),      # 已删除字段（通过valid_begin_time、valid_end_time判断）
            "status": qa_item.get("status", 1),                     # 启用状态（0禁用 1启用）
            "category": qa_item.get("category_all_name", ""),       # 四级分类
            "valid_begin_time": qa_item.get("valid_begin_time"),
            "valid_end_time": qa_item.get("valid_end_time"), 
            "source": qa_item.get("source"),                        # 数据来源（1: "知识平台", 2: "其他"）
            "is_delete": qa_item.get("is_delete", 0),               # 删除状态（0未删除 1已删除）
        }
        answer_meta = qa_item.get("answer_list", [])

        # 处理主问题
        if main_question:
            doc = Document(
                id=qa_item.get("question_id"),  # 主键id，避免重复
                content=main_question,          # 匹配标准问
                meta={
                    **question_meta,
                    "answers": answer_meta,  
                    "is_main_question": True
                }
            )
            documents.append(doc)

        # 处理相似问题（每个相似问题作为独立Document，但共享同一套答案）
        for similar in qa_item.get("similar_question_list", []):
            similar_question = similar.get("similar_question", "")
            question_meta["is_delete"] = similar.get("is_delete", 0)  # 11.19 update:覆盖更新is_delete
            if similar_question:
                doc = Document(
                    id=similar.get("similar_id"),  # 主键id，避免重复
                    content=similar_question,      # 匹配相似问
                    meta={
                        **question_meta,
                        "answers": answer_meta,
                        "similar_id": similar.get("similar_id"),    # 增加一个新的相似问题ID字段
                        "is_main_question": False
                    }
                )
                documents.append(doc)

        return documents

    def load_data(self, data_path_list: List[str]) -> List[Document]:
        """读取 JSON 格式 QA 数据并转换为 Document 对象"""
        data = []
        for data_path in data_path_list:
            with open(data_path, "r", encoding="utf-8") as file:
                data.extend(json.load(file))

        docs = []
        for qa_item in data:
            docs.extend(self._convert_qa_to_document(qa_item))

        logger.info(f"{len(data)} QA items loaded. {len(docs)} documents created.")
        return docs


class FAQ:
    """FAQ 检索系统"""
    def __init__(self, qa_path_list: List[str], model_type="stella", is_whitening=True):
        self.recall_module = Recall(
            self._get_docs(qa_path_list),
            model_type=model_type,
            is_whitening=is_whitening
        )
        self.rank_module = GLOBAL_RANK_MODULE
    
    @staticmethod
    def _get_docs(qa_path_list):
        preprocessor = DataPreprocessor()
        docs = preprocessor.load_data(qa_path_list)
        # 将 docs 中所有 id 替换为索引
        for i, doc in enumerate(docs):
            doc.id = i
        return docs

    def search(self, query, size=5, search_strategy='hybrid'):
        """执行 FAQ 检索"""
        if isinstance(query, str):
            query = [query]

        try:
            recall_hits = self.recall_module.retrieve(query, topK=size, search_strategy=search_strategy)
            if search_strategy == "hybrid":
                results = self.rank_module.rerank(query, recall_hits[0], recall_hits[1], fusion=True)
            else:
                results = recall_hits
        except Exception as e:
            logger.error(f"检索过程中发生错误：{e}")
            traceback.print_exc()
            results = [[]]

        return results[0]  # 返回单 query 结果


if __name__ == '__main__':
    # 需要替换为实际文件路径（QA 数据 JSON 文件）
    qa_paths = ["./data/faq1.json", "./data/faq2.json"]
    faq_sys = FAQ(qa_paths, model_type="stella", is_whitening=True)

    # 调试接口
    top_k = 5
    while True:
        query = input("Enter query: ")
        if query.lower() in ["exit", "quit"]:
            print("退出程序。")
            break
        try:
            start_time = time.time()

            results_embed = faq_sys.search(query, size=top_k, search_strategy='embedding')
            print("Embedding 检索：", [[res["content"], res["score"]] for res in results_embed[:top_k]])

            results_hybrid = faq_sys.search(query, size=top_k, search_strategy='hybrid')
            print("混合检索：", [[res["content"], res["score"]] for res in results_hybrid[:top_k]])

            end_time = time.time()
            print("检索耗时：", round(end_time - start_time, 3), "秒")

        except Exception as e:
            print(f"检索过程出错：{e}")
            traceback.print_exc()