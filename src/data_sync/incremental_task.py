import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable

import requests
import pandas as pd
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

"""
通用「增量更新」轮询任务（DB → 增量文件 → 增量接口）
•  通过「上次更新时间」找出有变更的主键 ID；  
•  再查详情，组合成结构化列表；  
•  写入增量 JSON 文件；  
•  调用增量更新接口（例如向量索引增量）；  
•  成功后更新 last_update_time，继续轮询。
"""

# ================== 配置 ==================
TIMEZONE = ZoneInfo("Asia/Shanghai")

UPDATE_INTERVAL = 20              # 轮询间隔（秒）
QUERY_TIMEOUT = 120               # 单次查询逻辑超时时间
THREAD_POOL_SIZE = 4
REQUEST_TIMEOUT = (10, 300)       # (连接超时, 读取超时)
MAX_RETRIES = 3

INCREMENT_FILE = "./data/increment_docs.json"
INCREMENTAL_UPDATE_URL = "http://your-service/increment-update"

DB_DSN = "mysql+pymysql://user:pwd@host:3306/dbname"


def start_of_today_str() -> str:
    now = datetime.now(TIMEZONE)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def now_str() -> str:
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class QueryResult:
    # 一条最终文档的数据结构，可按业务修改字段
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]


class IncrementalQuery:
    """
    通用的“分步查询 + 组合”的增量查询器
    - step1：根据 last_update_time 找出变更主键列表
    - step2+：按主键列表分步查详情并组合
    """

    def __init__(self, engine):
        self.engine = engine
        self.executor = ThreadPoolExecutor(
            max_workers=THREAD_POOL_SIZE, thread_name_prefix="increment_query"
        )

    # ------ 工具 ------
    def _read_sql(self, sql: str, params: Tuple = ()) -> pd.DataFrame:
        with self.engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        df.drop_duplicates(inplace=True)
        df.fillna("", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------ Step 1: 找出有变更的主键 ID 列表 ------
    def get_changed_ids(self, last_update_time: str) -> List[str]:
        """
        只保留结构：具体 SQL 按你的业务表写。
        假设业务通过多张表 update_time > last_update_time 来判断“有变更”。
        """
        sql = """
        -- 根据业务表结构改写
        SELECT DISTINCT main_id
        FROM main_table
        WHERE update_time > %s
           OR EXISTS (SELECT 1 FROM related_table_1 r1
                      WHERE r1.main_id = main_table.main_id
                        AND r1.update_time > %s)
           OR EXISTS (SELECT 1 FROM related_table_2 r2
                      WHERE r2.main_id = main_table.main_id
                        AND r2.update_time > %s)
        """
        df = self._read_sql(sql, (last_update_time, last_update_time, last_update_time))
        return [str(row["main_id"]) for _, row in df.iterrows()]

    # ------ Step 2: 主表信息 ------
    def get_main_info(self, ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        id_list = "','".join(ids)
        sql = f"""
        SELECT main_id, title, content, status, is_deleted
        FROM main_table
        WHERE main_id IN ('{id_list}')
        """
        df = self._read_sql(sql)
        return {str(row["main_id"]): row.to_dict() for _, row in df.iterrows()}

    # ------ Step 3: 关联信息示例（标签/相似问/答案等） ------
    def get_tags(self, ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not ids:
            return {}
        id_list = "','".join(ids)
        sql = f"""
        SELECT main_id,
               JSON_ARRAYAGG(
                   JSON_OBJECT(
                       'tag_id', tag_id,
                       'tag_name', tag_name
                   )
               ) AS tags
        FROM main_table_tags
        WHERE main_id IN ('{id_list}')
        GROUP BY main_id
        """
        df = self._read_sql(sql)
        return {
            str(row["main_id"]): json.loads(row["tags"])
            for _, row in df.iterrows()
        }

    def get_similar_items(self, ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        if not ids:
            return {}
        id_list = "','".join(ids)
        sql = f"""
        SELECT main_id,
               JSON_ARRAYAGG(
                   JSON_OBJECT(
                       'similar_id', similar_id,
                       'text', similar_text,
                       'is_deleted', is_deleted
                   )
               ) AS similar_list
        FROM main_table_similar
        WHERE main_id IN ('{id_list}')
        GROUP BY main_id
        """
        df = self._read_sql(sql)
        return {
            str(row["main_id"]): json.loads(row["similar_list"])
            for _, row in df.iterrows()
        }

    # ------ Step N: 组合最终文档 ------
    def combine_increment(self, last_update_time: str) -> List[QueryResult]:
        try:
            # Step 1：获取需要更新的主键
            ids = self.executor.submit(
                self.get_changed_ids, last_update_time
            ).result(timeout=QUERY_TIMEOUT)

            if not ids:
                return []

            # Step 2+：并行获取子信息
            futures = {
                "main": self.executor.submit(self.get_main_info, ids),
                "tags": self.executor.submit(self.get_tags, ids),
                "similar": self.executor.submit(self.get_similar_items, ids),
            }
            results = {
                key: fut.result(timeout=QUERY_TIMEOUT)
                for key, fut in futures.items()
            }

            main_info = results["main"]
            tags_map = results["tags"]
            similar_map = results["similar"]

            docs: List[QueryResult] = []
            for mid in ids:
                if mid not in main_info:
                    continue
                base = main_info[mid]
                if base.get("is_deleted"):  # 按需要处理逻辑删除
                    status = "deleted"
                else:
                    status = "active"

                doc = QueryResult(
                    id=mid,
                    title=base.get("title", ""),
                    content=base.get("content", ""),
                    metadata={
                        "status": status,
                        "raw_status": base.get("status"),
                        "tags": tags_map.get(mid, []),
                        "similar_items": similar_map.get(mid, []),
                    },
                )
                docs.append(doc)

            return docs

        except Exception:
            traceback.print_exc()
            return []

    def close(self):
        self.executor.shutdown(wait=False)
        self.engine.dispose()


# ================== 主循环：写增量文件 + 调用增量接口 ==================
def write_increment_file(docs: List[QueryResult], path: str) -> None:
    payload = [
        {
            "id": d.id,
            "title": d.title,
            "content": d.content,
            "metadata": d.metadata,
        }
        for d in docs
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"{len(docs)} docs written to {path}")


def call_incremental_update(url: str, max_retries: int = MAX_RETRIES) -> None:
    print("Call incremental update:", url)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                print("Incremental update success:", resp.text)
                return
            else:
                print(f"Incremental update failed, status={resp.status_code}")
        except Exception as e:
            print(f"Incremental update request error: {e}")
    print("Incremental update finished with failure after retries")


def main():
    last_update_time = start_of_today_str()
    engine = create_engine(DB_DSN)
    iq = IncrementalQuery(engine)

    print(f"Incremental sync start, initial last_update_time={last_update_time}")
    print(f"Interval = {UPDATE_INTERVAL} sec")

    try:
        while True:
            try:
                start_ts = time.time()
                docs = iq.combine_increment(last_update_time)
                print(f"Query cost: {time.time() - start_ts:.3f}s")

                if not docs:
                    print(f"{now_str()} no increments")
                else:
                    print("=" * 50)
                    print(f"{now_str()} found {len(docs)} increment docs")
                    last_update_time = now_str()

                    # 写增量文件
                    write_increment_file(docs, INCREMENT_FILE)

                    # 调用增量向量/索引更新接口
                    call_incremental_update(INCREMENTAL_UPDATE_URL)
                    print("=" * 50)

                time.sleep(UPDATE_INTERVAL)

            except Exception:
                print(f"{now_str()} error in main loop")
                traceback.print_exc()
                time.sleep(20)
    finally:
        iq.close()
        print("Program exit")


if __name__ == "__main__":
    main()