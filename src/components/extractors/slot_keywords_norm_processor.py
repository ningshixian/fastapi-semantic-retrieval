import os
import json
import logging
from flashtext import KeywordProcessor


class SlotKeywordsNormProcessor(object):
    def __init__(self, slot_keywords_path: str):
        # 初始化关键词处理器
        self.slot_keywords_processor = SlotKeywordsNormProcessor.__init_slot_keywords_processor(slot_keywords_path)

    @staticmethod
    def __load_json_dict(slot_keywords_file_path):
        """
        加载归一化关键词数据
        :param slot_keywords_file_path: 关键词归一化配置文件路径（json文件，key为归一化名称，value为关键词列表）
        :return: dict: {归一化名称: [关键词1, 关键词2, ...]}
        """
        with open(slot_keywords_file_path, 'r', encoding='utf-8') as read_file:
            return json.load(read_file)

    @staticmethod
    def __init_slot_keywords_processor(slot_keywords_file_path):
        """
        初始化归一化关键词处理器
        :param slot_keywords_file_path: 关键词归一化配置文件路径
        :return: KeywordProcessor 对象
        """
        if not slot_keywords_file_path:
            return None
        keyword_dict = SlotKeywordsNormProcessor.__load_json_dict(slot_keywords_file_path)

        # 可在此添加额外的归一化关键词映射（业务可自定义）
        # keyword_dict['通用类别1'] = ["词A", "词B", "词C"]
        # keyword_dict['通用类别2'] = ["词X", "词Y"]

        # 创建关键字归一化处理器（忽略大小写匹配）
        keyword_normalization_processor = KeywordProcessor(case_sensitive=False)
        keyword_normalization_processor.add_keywords_from_dict(keyword_dict)
        return keyword_normalization_processor

    def slot_filling(self, text):
        """
        槽位提取
        :param text: 待抽取文本
        :return: list: 抽取结果，如 [(归一化名称, 起始位置, 结束位置), ...]
        """
        return self.slot_keywords_processor.extract_keywords(text, span_info=True)


if __name__ == '__main__':
    # 示例用法（slot.json 文件中已包含基础映射）
    slot_processor = SlotKeywordsNormProcessor('./data_factory/slot.json')

    # 可在运行时动态添加关键词映射
    slot_processor.slot_keywords_processor.add_keywords_from_dict({
        "类别A": ["关键词1", "关键词2", "关键词3"],
        "类别B": ["关键词4", "关键词5"],
        "类别C": ["关键词6", "关键词7"]
    })

    print(slot_processor.slot_filling('这里输入一段待匹配的文本'))
    # 示例输出: [('类别A', 位置1, 位置2), ('类别B', 位置3, 位置4)]