"""
translate_pipeline.py
=====================
Pipeline 3: 英文 → 繁体中文翻译

任务:把英文菜名翻译成繁体中文,方便香港用户阅读

实现方式:两步走
  Step 1: 用 Helsinki-NLP/opus-mt-en-zh 翻译英文到简体中文
  Step 2: 用 OpenCC 把简体中文转繁体中文(香港繁体)

⚠️ 重要说明:
  本模块不使用 transformers 的 pipeline() 高层函数,而是直接用
  AutoTokenizer + AutoModelForSeq2SeqLM,原因是:
    - transformers v5 移除了 "translation" 这个 pipeline task,
      会导致 pipeline("translation", ...) 抛 KeyError
    - 用底层 API 更稳定,不受 pipeline task 名变化影响
    - 本质仍是一个完整的 inference pipeline:
        tokenize → encode → generate → decode
      报告中仍可视为一个 HuggingFace pipeline
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
    加载并缓存英中翻译的 tokenizer 和 model。

    不使用 pipeline() 高层函数,而是直接用 AutoTokenizer + AutoModelForSeq2SeqLM,
    避开 transformers 版本变化导致的 pipeline task 名问题。

    返回:
        (tokenizer, model) 元组
    """
    # tokenizer 负责把英文文本切分成模型可识别的 token id
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATE_MODEL_NAME)

    # AutoModelForSeq2SeqLM 是 seq2seq 任务的通用接口,
    # 对 opus-mt 系列会自动加载 MarianMTModel
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL_NAME)

    # 设置为评估模式(关闭 dropout)
    model.eval()

    return tokenizer, model


@st.cache_resource
def load_opencc_converter():
    """
    加载并缓存 OpenCC 转换器(简体→香港繁体)。

    OpenCC 是基于字符映射表的工具,几乎不占内存,
    但仍然缓存它以避免重复初始化的开销。
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
        1. 用 tokenizer 把英文文本编码成 input_ids
        2. 用 model.generate() 做自回归生成,得到中文 token id
        3. 用 tokenizer.decode() 解码成简体中文字符串
        4. 用 OpenCC 把简体中文转换成香港繁体中文
    """
    # 处理空输入,直接返回空字符串
    if not english_text or not english_text.strip():
        return ""

    # 加载模型(已缓存)
    tokenizer, model = load_translator()
    converter = load_opencc_converter()

    # 步骤 1:把英文文本编码成模型输入张量
    # return_tensors="pt" 表示返回 PyTorch 张量
    # truncation=True 防止过长输入溢出模型最大长度
    inputs = tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # 步骤 2:用 no_grad() 节省内存(推理不需要梯度)
    with torch.no_grad():
        # generate() 会自回归地生成 token 序列
        # max_length 限制输出长度
        # num_beams=4 用 beam search,翻译质量更好(但稍慢)
        output_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    # 步骤 3:把 token id 解码成简体中文字符串
    # skip_special_tokens=True 跳过 [CLS] [SEP] 等特殊 token
    simplified_zh = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 步骤 4:简体 → 繁体(香港)
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
        这里采用顺序调用而不是真正的 batching,
        是因为菜名通常不多(3-10 道),没必要为此处理 padding 等复杂逻辑。
    """
    return [translate_to_traditional_chinese(text) for text in english_texts]
