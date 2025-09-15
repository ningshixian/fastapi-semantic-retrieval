"""
FastAPI application module (generalized version)
"""
import os
import sys
import inspect
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware

# ✅ 路由模块（泛化名称）
from src.api.api import router as api_router  

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

def get_application() -> FastAPI:
    """Create and configure a FastAPI application."""
    dependencies = []

    application = FastAPI(
        title="Knowledge Semantic Search API",
        description="A semantic search service for knowledge-based question answering",
        dependencies=dependencies if dependencies else None
    )

    # CORS settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API router
    api_prefix: str = "/api"
    application.include_router(api_router, prefix=api_prefix)

    return application


# ===== 全局访问方法 =====
def get():
    """Returns the global main service instance."""
    return INSTANCE

def get_main_service():
    """Returns the main FAQ/Q&A instance."""
    return main_service

def get_recommendation_service():
    """Returns the recommendation FAQ instance."""
    return recommendation_service

def get_entity_extractor():
    """Returns the entity extraction instance."""
    return entity_extractor


# ===== 初始化应用 =====
app = get_application()

# ===== 加载核心服务（具体路径已泛化） =====
main_service = FAQ([
    paths.qa.main_source, 
    paths.qa.secondary_source, 
    paths.qa.greeting_source
])
recommendation_service = FAQ([
    paths.qa.common_source
])
INSTANCE = main_service

# ===== 加载实体识别模块 =====
entity_extractor = EntityExtractor(
    paths.entity.slot_model, 
    paths.entity.entity_model
)
# entity_extractor.entity_extract(text, "entityType", {})


if __name__ == "__main__":
    # 显示所有路由
    for route in app.routes:
        if hasattr(route, "methods"):
            print({"path": route.path, "name": route.name, "methods": route.methods})

    import uvicorn
    uvicorn.run(
        app="application:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=1
    )