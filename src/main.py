"""
FastAPI application module (generalized version)
"""
import os
import sys
import json
import inspect
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

# ✅ 路由模块（泛化名称）
from api.api import router as api_router  

# ✅ 配置路径（泛化）
from configs.config import paths  

# ✅ 模块依赖（名称已泛化）
from src.components.extractors.ner import EntityExtractor
from src.components.faq import FAQ

"""
示例启动命令 (可根据实际情况修改)
------------------------------------------------
export APP_ENV=test
export CUDA_VISIBLE_DEVICES=0

# 启动方式 1 - uvicorn
nohup uvicorn "application:app" --host 0.0.0.0 --port 8000 --workers 1 > logs/app.log 2>&1 &

# 启动方式 2 - gunicorn
nohup gunicorn application:app -c configs/gunicorn_config_api.py > logs/app.log 2>&1 &
------------------------------------------------
"""


# 提供全局状态管理
# 重要组件都可以作为全局单例访问，避免了重复初始化的开销
class AppState:
    def __init__(self):
        self.INSTANCE = None
        self.main_faq = None
        self.recommendation_faq = None
        self.voice_faq = None
        self.uie = None
        self.intent_dict = {}
        self.param_dict = {}
        self.entity_dict = {}


app_state = AppState()


# ------------------------------
# Lifespan Manager
# ------------------------------
# async def lifespan(app: FastAPI):
def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: init resources here.
    """
    # 初始化 FAQ 系统
    app_state.recommendation_faq = FAQ([paths.qa.common])
    app_state.main_faq = FAQ([paths.qa.robot, paths.kafka.onetouch, paths.qa.greeting])
    app_state.INSTANCE = app_state.main_faq
    # 初始化 NER 模块
    app_state.uie = EntityExtractor(paths.kafka.slot, paths.entities.car)
    # 加载 Redis 快照
    try:
        # 创建Redis客户端
        from utils import RedisUtilsSentinel
        from configs.config import redis_config, snapshot_key
        redis_client = RedisUtilsSentinel(redis_config.__dict__)
        snap = json.loads(redis_client.get(snapshot_key))
        app_state.intent_dict = snap["data"].get("intent2param", {})
        app_state.param_dict = snap["data"].get("param2ent", {})
        app_state.entity_dict = snap["data"].get("ent2vocab", {})
        logger.info("Loaded snapshot from Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to load snapshot from Redis: {e}")

    # 自动注册路由
    # API router
    api_prefix: str = "/api"
    app.include_router(api_router, prefix=api_prefix)

    yield


def create_app() -> FastAPI:
    """Create and configure a FastAPI application."""
    dependencies = []

    application = FastAPI(
        title="Knowledge Semantic Search API",
        description="A semantic search service for knowledge-based question answering",
        dependencies=dependencies if dependencies else None, 
        lifespan=lifespan,
        version="1.0.0"
    )

    # CORS settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


# ===== 全局访问方法 =====
def get_instance() -> FAQ:
    return app_state.INSTANCE

def get_main_faq() -> FAQ:
    """获取主FAQ实例"""
    return app_state.main_faq

def get_uie() -> EntityExtractor:
    """获取实体识别实例"""
    return app_state.uie

def set_uie(_uie):
    app_state.uie = _uie


# ===== 初始化应用 =====
app = create_app()


if __name__ == "__main__":
    # 显示所有路由
    for route in app.routes:
        if hasattr(route, "methods"):
            print({"path": route.path, "name": route.name, "methods": route.methods})

    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        workers=1
    )