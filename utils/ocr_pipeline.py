"""
ocr_pipeline.py
===============
Pipeline 1: 图片转文字 (Image-to-Text / OCR)

任务:从英文菜单图片中提取菜名文本

模型:microsoft/trocr-small-printed
  - TrOCR 是一个 encoder-decoder 架构的 OCR 模型
  - encoder 是 Vision Transformer,decoder 是 Text Transformer
  - 选 small 版而不是 base 版,是为了适配 Streamlit Cloud 1GB 内存限制
  - 注意:TrOCR 一次只能识别一行印刷文本(single-line OCR),
    所以建议用户上传裁剪过的单道菜区域

注意事项:
  - 用 @st.cache_resource 缓存模型,避免每次请求都重新加载
  - 模型加载需要约 250MB 内存
"""

import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


# OCR 模型的 Hugging Face 名称
# 如果想要更高准确率(但内存允许),可以换成 "microsoft/trocr-base-printed"
OCR_MODEL_NAME = "microsoft/trocr-small-printed"


@st.cache_resource(show_spinner="正在加载 OCR 模型(首次加载约需 30 秒)...")
def load_ocr_model():
    """
    加载并缓存 TrOCR 模型。

    使用 Streamlit 的 @st.cache_resource 装饰器,
    确保整个 app 生命周期内模型只被加载一次,
    避免每次用户上传图片都重新下载/加载。

    返回:
        (processor, model) 元组
    """
    # processor 负责把图像转换成模型可接收的 pixel_values
    processor = TrOCRProcessor.from_pretrained(OCR_MODEL_NAME)

    # 加载 VisionEncoderDecoderModel(ViT encoder + Transformer decoder)
    model = VisionEncoderDecoderModel.from_pretrained(OCR_MODEL_NAME)

    # 设置为评估模式(关闭 dropout 等训练专用层)
    model.eval()

    return processor, model


def run_ocr(image: Image.Image) -> str:
    """
    对一张图片做 OCR,返回识别出的英文文本。

    参数:
        image: PIL.Image 对象(用户上传的菜单图片)

    返回:
        识别出的英文文本字符串(已去除首尾空白)

    流程:
        1. 把图片统一转成 RGB 模式(TrOCR 要求 3 通道)
        2. 用 processor 把图像编码成 pixel_values 张量
        3. 用 model.generate() 做自回归式的文本生成(decoder 一个 token 一个 token 生成)
        4. 用 processor 把 token id 解码成可读的字符串
    """
    # 加载模型(已缓存)
    processor, model = load_ocr_model()

    # 步骤 1:确保图片是 RGB 模式
    # 有些 PNG 带 alpha 通道(RGBA),有些是灰度图(L),统一转换避免出错
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 步骤 2:把 PIL 图片转换成模型输入张量
    # return_tensors="pt" 表示返回 PyTorch 张量
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 步骤 3:用 no_grad() 节省内存(推理时不需要计算梯度)
    with torch.no_grad():
        # generate() 会自回归地生成 token 序列
        # max_length=128 限制最长输出长度,菜名通常不会太长
        generated_ids = model.generate(pixel_values, max_length=128)

    # 步骤 4:把 token id 转换回可读文本
    # skip_special_tokens=True 表示跳过 [CLS] [SEP] 等特殊 token
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return generated_text.strip()


def split_into_dishes(text: str) -> list:
    """
    把 OCR 识别出的多行文本按行切分成多道菜。

    因为用户可能上传带多道菜的图片,即使 TrOCR 主要支持单行,
    我们也尝试按换行符切分,过滤掉太短的噪声行。

    参数:
        text: OCR 输出的原始文本

    返回:
        list[str]:切分后的菜名列表
    """
    # 按换行符切分
    lines = text.split("\n")

    # 过滤:去除首尾空白,丢弃长度过短的行(通常是噪声)
    dishes = [line.strip() for line in lines if len(line.strip()) >= 3]

    # 如果切分结果为空,把原始文本当作一道菜
    if not dishes:
        dishes = [text.strip()] if text.strip() else []

    return dishes
