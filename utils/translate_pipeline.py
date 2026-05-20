"""
translate_pipeline.py
=====================
Pipeline 3: 英文 → 繁体中文翻译

任务:把英文菜名翻译成繁体中文,方便香港用户阅读

实现方式:两步走
  Step 1: 用 Helsinki-NLP/opus-mt-en-zh 翻译英文到简体中文
  Step 2: 用 OpenCC 把简体中文转繁体中文(香港繁体)

为什么不直接英→繁:
  - HF 上没有专门的英→繁中模型
  - 直接用 NLLB-200 可以输出繁体,但模型太大(600MB+),Streamlit Cloud 装不下
  - opus-mt-en-zh 模型小(~300MB),OpenCC 是纯字符映射几乎不占资源
"""

import streamlit as st
from transformers import pipeline
from opencc import OpenCC


# 英文 → 中文翻译模型
TRANSLATE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

# OpenCC 配置:s2hk 表示 Simplified Chinese → Hong Kong Traditional Chinese
# 其他可选:s2t (繁体)、s2tw (台湾繁体)
# 香港用户用 s2hk 最合适(包含粤语习惯用字)
OPENCC_CONFIG = "s2hk"


@st.cache_resource(show_spinner="正在加载翻译模型(首次加载约需 20 秒)...")
def load_translator():
    """
    加载并缓存英中翻译 pipeline。

    返回:
        translator pipeline 对象
    """
    translator = pipeline(
        task="translation",
        model=TRANSLATE_MODEL_NAME,
        device=-1,   # CPU 模式
    )
    return translator


@st.cache_resource
def load_opencc_converter():
    """
    加载并缓存 OpenCC 转换器(简体→香港繁体)。

    OpenCC 是基于字符映射表的工具,几乎不占内存,
    但我们仍然缓存它以避免重复初始化的开销。
    """
    return OpenCC(OPENCC_CONFIG)


def translate_to_traditional_chinese(english_text: str) -> str:
    """
    把英文文本翻译成繁体中文(香港地区用字)。

    参数:
        english_text: 待翻译的英文文本(通常是菜名)

    返回:
        繁体中文翻译结果字符串

    流程:
        1. 用 opus-mt 把英文翻译成简体中文
        2. 用 OpenCC 把简体中文转换成香港繁体中文
    """
    # 处理空输入,直接返回空字符串
    if not english_text or not english_text.strip():
        return ""

    # 加载模型(已缓存)
    translator = load_translator()
    converter = load_opencc_converter()

    # 步骤 1:英文 → 简体中文
    # max_length 限制输出长度,避免无限循环
    result = translator(english_text, max_length=128)

    # pipeline 输出格式:[{"translation_text": "..."}]
    simplified_zh = result[0]["translation_text"]

    # 步骤 2:简体 → 繁体(香港)
    traditional_zh = converter.convert(simplified_zh)

    return traditional_zh.strip()


def translate_batch(english_texts: list) -> list:
    """
    批量翻译多道菜名,返回繁体中文列表。

    参数:
        english_texts: list[str],英文菜名列表

    返回:
        list[str],对应的繁体中文菜名列表(顺序与输入一致)

    说明:
        这里采用顺序调用而不是 pipeline 的真正 batching,
        是因为菜名通常不多(3-10 道),没必要为此处理 padding 等复杂逻辑。
    """
    return [translate_to_traditional_chinese(text) for text in english_texts]
