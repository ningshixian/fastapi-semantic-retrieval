"""
FastAPI application module
"""
import os
import sys
import inspect
from fastapi import APIRouter, Depends, FastAPI


def apirouters():
    """
    输出所有available路由
    """

    # # Get handle to api module
    # api = sys.modules[".".join(__name__.split(".")[:-1])]
    # # api = sys.modules["faq-semantic-retrieval.routers"]

    # 尝试导入指定模块
    import importlib
    api = importlib.import_module("routers")

    available = {}
    for name, rclass in inspect.getmembers(api, inspect.ismodule):
        if hasattr(rclass, "router") and isinstance(rclass.router, APIRouter):
            available[name.lower()] = rclass.router

    # print("routers: ", available)
    return available


# Load YAML settings (这个配置可以用于控制注册那些路由！)
config = {
    "embeddings": None, 
    "entity": None, 
    "similarity": None, 
    "faq_search_engine": None, 
    "faq_recommend_engine": None, 
    "update_api": None, 
    "rag": None, 
    "": None, 
}

# Conditionally add routes based on configuration
router = APIRouter()
for name, router in apirouters().items():
    if name in config:
        router.include_router(router)  # 注册路由
