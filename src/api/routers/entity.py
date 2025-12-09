from typing import List
from fastapi import APIRouter, Body, Request

from schemas.response import response_code
from src.main import app as application

router = APIRouter()


# 这里模拟一个实体抽取函数（实际需要替换成你自己的模型或算法）
def entity_extract_from_text(text, entity_mode: str = "default"):
    """
    通用实体抽取函数
    """
    flag, entity_value, start_idx, end_idx = application.get_uie().entity_extract(
        text, entity_mode, entity={'entityName':''}
    )
    return flag, entity_value, start_idx, end_idx


@router.post("/entity_extract", summary="实体抽取", name="实体抽取")
def entity_extract(
    request: Request,
    text: str = Body(..., embed=True)
):
    """
    通用实体抽取接口
    """
    # 保证输入是列表形式
    if isinstance(text, str):
        text = [text]
    
    # 调用通用实体抽取方法
    flag, entity_value, start_idx, end_idx = entity_extract_from_text(
        text, entity_mode="default"
    )

    match_entity: List[dict] = []
    if flag:
        match_entity.append({
            'start': start_idx,            # 实体开始位置
            'end': end_idx,                # 实体结束位置
            'entity_name': "",              # 实体名称，可为空
            'entity_value': entity_value,   # 实体值
            'entity_mode': "defaultEntity", # 实体类型，更通用的命名
            'param_name': "",               # 参数名
            "prebuild": False,              # 是否系统预置实体
            "is_prebuild": False            # 是否系统预置实体
        })

    return response_code.resp_200(data=match_entity)


@router.post("/ner", summary="算法实体抽取", name="算法实体抽取")
def ner(
    request: Request, 
    text: str=Body(..., embed=True),
    entity_name: str=Body(default="", embed=True), # 实体名称,对应七鱼的entityType
    param_name: str=Body(default="", embed=True),  # 变量名称，与实体名称一一对应
):
    if isinstance(text, str):
        text = [text]

    # algorithmEntity 目前映射车型提取，返回提取的第一个结果
    flag, entity_value, start_idx, end_idx = application.get_uie().entity_extract(
        text, 
        entity_mode="algorithmEntity", 
        entity={
            "entityName": entity_name,
            "entityType": 5,
            "vocab": {}
    })

    match_entity = []
    if flag:
        match_entity.append({
            'start': start_idx,  #实体位置
            'end': end_idx,  #实体位置
            'entity_name': entity_name,  #实体名称,对应七鱼的entityType
            'entity_value': entity_value,  #实体值
            'entity_mode': "algorithmEntity",  #实体类型（0-系统/1-枚举/2-正则/3-意图/4-其他/5-算法）
            'param_name': param_name,  #变量名称，与实体名称一一对应
            # 'flow_name': intent.get('flowName', '')  #关联流程名称
            "prebuild": False,  # 是否系统预置实体
            "is_prebuild": False  # 是否系统预置实体
        })
    return response_code.resp_200(data=match_entity)
