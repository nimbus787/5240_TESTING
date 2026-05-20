"""
allergen_pipeline.py
====================
Pipeline 2: 过敏原识别 (Multi-label Allergen Detection)

支持两种检测模式:
  1. Zero-shot Classification:
     - 模型:valhalla/distilbart-mnli-12-3
     - 原理:把过敏原识别转换成自然语言推断 (NLI) 问题
     - 优点:无需训练,泛化能力强,符合 deep learning pipeline 要求
     - 缺点:推理较慢(每道菜要跑 14 次 NLI 推断)

  2. Keyword Rule-based:
     - 原理:用预定义的关键词字典做字符串匹配
     - 优点:极快,可解释,作为 baseline
     - 缺点:无法识别同义词、生僻菜名

在报告里可以把两种模式做对比实验,凸显 deep learning 的价值。
"""

import re
import streamlit as st
from transformers import pipeline

# 从 config 模块导入 14 类过敏原标签和关键词字典
from utils.config import (
    ALLERGEN_LABELS,
    ALLERGEN_KEYWORDS,
    get_risk_level,
)


# Zero-shot 分类模型名称
# distilbart-mnli-12-3 是 BART-MNLI 的蒸馏小版本,约 530MB
# 如果想要更高准确率(且内存允许),可换成 "facebook/bart-large-mnli"(约 1.6GB)
ZSC_MODEL_NAME = "valhalla/distilbart-mnli-12-3"


# ===========================================================================
# 模式 A: Zero-shot Classification 模式
# ===========================================================================

@st.cache_resource(show_spinner="正在加载零样本分类模型(首次加载约需 30 秒)...")
def load_zsc_pipeline():
    """
    加载并缓存 zero-shot classification pipeline。

    Hugging Face 的 pipeline() 是一个高层封装,
    内部会自动处理 tokenizer、model、postprocessing 三个步骤。

    返回:
        transformers.Pipeline 对象,可直接调用
    """
    # device=-1 表示用 CPU(Streamlit Cloud 没有 GPU)
    # 如果在本地有 GPU,可以改成 device=0
    zsc_pipe = pipeline(
        task="zero-shot-classification",
        model=ZSC_MODEL_NAME,
        device=-1,
    )
    return zsc_pipe


def detect_allergens_zsc(dish_text: str, threshold: float = 0.5) -> list:
    """
    用 Zero-shot Classification 模型检测一道菜可能含有的过敏原。

    参数:
        dish_text: 一道菜的英文名称或描述
        threshold: 置信度阈值,大于该值的标签会被返回

    返回:
        list of dict:[{"label": "fish", "score": 0.92, "risk": "high"}, ...]
        按置信度从高到低排序
    """
    if not dish_text or not dish_text.strip():
        return []

    # 加载已缓存的模型
    classifier = load_zsc_pipeline()

    # 调用 zero-shot classifier
    # multi_label=True 是关键:让每个标签的概率独立计算(每个 label 独立做二分类)
    # 而不是 multi-class(所有 label 总和为 1)
    # 因为一道菜可能同时含多个过敏原,所以必须用 multi_label
    result = classifier(
        dish_text,
        candidate_labels=ALLERGEN_LABELS,
        # hypothesis_template 控制如何构造 NLI 假设
        # 默认是 "This example is {}.",这里我们换一个更贴合食物场景的模板
        hypothesis_template="This dish contains {}.",
        multi_label=True,
    )

    # result 的结构:
    # {
    #   "sequence": "Grilled salmon with butter sauce",
    #   "labels":   ["fish", "milk", "gluten", ...],  按概率降序
    #   "scores":   [0.95, 0.87, 0.12, ...],
    # }

    # 把超过阈值的过敏原整理成结果列表
    detected = []
    for label, score in zip(result["labels"], result["scores"]):
        if score >= threshold:
            detected.append({
                "label": label,
                "score": float(score),
                "risk":  get_risk_level(score),
            })

    return detected


# ===========================================================================
# 模式 B: 关键词规则模式(baseline)
# ===========================================================================

def detect_allergens_keyword(dish_text: str) -> list:
    """
    用关键词字典匹配检测一道菜可能含有的过敏原。

    这是一个 rule-based baseline,作为零样本模型的对比。
    在报告的实验章节中,可以对比两种方法的:
      - 准确率(Micro-F1 / Macro-F1)
      - 运行时间(单条样本耗时)
      - 召回率(food safety 场景下尤为重要)

    参数:
        dish_text: 一道菜的英文名称或描述

    返回:
        list of dict:格式同 detect_allergens_zsc,但 score 是 1.0(命中即满分)
    """
    if not dish_text or not dish_text.strip():
        return []

    # 统一转小写,关键词字典里也都是小写
    text_lower = dish_text.lower()

    detected = []

    # 遍历 14 类过敏原,检查每个的关键词是否在文本中出现
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        # 对每个关键词做"完整单词"匹配
        # 用 \b 单词边界避免误判(例如 "buttery" 不应匹配 "butter")
        # 但这里"butter"在"buttery"里其实算合理匹配,所以我们用更宽松的 substring 匹配
        # 同时记录命中的具体关键词,方便调试
        hit_keywords = []
        for kw in keywords:
            # 用正则的单词边界做匹配,避免 "egg" 误匹配 "eggplant"
            # 但对包含空格的多词关键词(如 "soy sauce"),直接 substring 匹配
            if " " in kw:
                if kw in text_lower:
                    hit_keywords.append(kw)
            else:
                # \b 表示单词边界,适合单个单词关键词
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text_lower):
                    hit_keywords.append(kw)

        # 只要至少命中一个关键词,就认定该过敏原存在
        if hit_keywords:
            detected.append({
                "label":         allergen,
                "score":         1.0,           # 关键词命中视为满分
                "risk":          "high",         # 关键词命中视为高风险
                "hit_keywords":  hit_keywords,  # 用于显示触发原因
            })

    return detected


# ===========================================================================
# 统一接口:根据模式选择调用哪个检测函数
# ===========================================================================

def detect_allergens(dish_text: str, mode: str = "zero-shot",
                     threshold: float = 0.5) -> list:
    """
    统一的过敏原检测入口,根据模式分发到对应的实现。

    参数:
        dish_text: 一道菜的英文名称或描述
        mode:      "zero-shot" 或 "keyword"
        threshold: zero-shot 模式下的置信度阈值

    返回:
        list of dict,见各模式函数的返回格式
    """
    if mode == "keyword":
        return detect_allergens_keyword(dish_text)
    else:
        # 默认用 zero-shot 模式
        return detect_allergens_zsc(dish_text, threshold=threshold)
