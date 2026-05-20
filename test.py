"""
app.py
======
SafeBite HK - UK Menu Allergy Assistant
主 Streamlit 应用程序

项目:HKUST ISOM5240 期末项目
公司:OpenRice
目标用户:赴英国旅游的香港居民

3 个 Pipeline 串联:
  Pipeline 1: OCR        (图片 → 英文菜名)
  Pipeline 2: Allergen   (英文菜名 → 14 类过敏原标签)
  Pipeline 3: Translate  (英文菜名 → 繁体中文菜名)

UI 设计:
  - 双 Tab 输入(图片上传 / 直接输入文本),方便测试和 demo
  - 侧边栏:模式切换(zero-shot vs keyword)+ 阈值调整
  - 结果区:菜名、过敏原、风险等级、置信度的综合表格
"""

import streamlit as st
from PIL import Image

# 导入自定义的 3 个 pipeline 模块
from utils.ocr_pipeline import run_ocr, split_into_dishes
from utils.allergen_pipeline import detect_allergens
from utils.translate_pipeline import translate_to_traditional_chinese

# 导入配置:过敏原中英对照、风险等级映射等
from utils.config import (
    ALLERGEN_ZH_MAP,
    RISK_ZH_MAP,
    RISK_EMOJI_MAP,
)


# ===========================================================================
# Streamlit 页面基本配置
# ===========================================================================

st.set_page_config(
    page_title="SafeBite HK - UK Menu Allergy Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# 页面顶部:标题和项目介绍
# ===========================================================================

st.title("🍽️ SafeBite HK")
st.subheader("UK Menu Allergy Assistant for OpenRice Travelers")

st.markdown(
    """
    **面向赴英港人的英文菜单過敏原識別助手** 

    上傳英國餐廳菜單照片或直接輸入菜名,系統會自動:
    1. 🔍 識別菜名 (OCR)
    2. ⚠️ 預測 14 類英國法定過敏原風險
    3. 🇭🇰 翻譯成繁體中文
    """
)

st.divider()


# ===========================================================================
# 侧边栏:用户可配置选项
# ===========================================================================

with st.sidebar:
    st.header("⚙️ 設定")

    # 选项 1:过敏原检测模式
    detection_mode = st.radio(
        "過敏原檢測模式",
        options=["zero-shot", "keyword"],
        format_func=lambda x: {
            "zero-shot": "🤖 Zero-shot 模型 (Deep Learning)",
            "keyword":   "📖 關鍵詞規則 (Baseline)",
        }[x],
        help="Zero-shot 用深度學習模型推理;關鍵詞規則用字典匹配做 baseline 對比。",
    )

    # 选项 2:置信度阈值(仅 zero-shot 模式可调)
    if detection_mode == "zero-shot":
        threshold = st.slider(
            "置信度閾值 (Threshold)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="只有預測概率超過此閾值的過敏原才會被報告。",
        )
    else:
        # 关键词模式没有阈值概念,设个占位值
        threshold = 0.5

    st.divider()

    # 项目信息
    st.markdown(
        """
        ### 📚 關於本項目

        - **公司**: OpenRice
        - **課程**: ISOM5240
        - **3 個 Hugging Face Pipelines**:
            1. TrOCR (image-to-text)
            2. DistilBART-MNLI (zero-shot)
            3. Opus-MT (en→zh) + OpenCC

        ### ⚠️ 免責聲明
        本工具僅提供 AI 過敏原風險提示,不能替代與餐廳人員的直接確認。
        食物安全請以實際標籤為準。
        """
    )


# ===========================================================================
# 核心处理函数:把一道菜走完 Pipeline 2 + Pipeline 3
# ===========================================================================

def process_single_dish(english_dish: str, mode: str, thresh: float) -> dict:
    """
    把单道菜的英文名跑完过敏原检测 + 翻译两个 pipeline。

    参数:
        english_dish: 英文菜名
        mode:         检测模式("zero-shot" / "keyword")
        thresh:       置信度阈值

    返回:
        dict 包含:
          - english:    英文菜名
          - chinese:    繁体中文菜名
          - allergens:  检测到的过敏原列表
    """
    # Pipeline 2: 检测过敏原
    allergens = detect_allergens(english_dish, mode=mode, threshold=thresh)

    # Pipeline 3: 翻译成繁体中文
    chinese = translate_to_traditional_chinese(english_dish)

    return {
        "english":   english_dish,
        "chinese":   chinese,
        "allergens": allergens,
    }


# ===========================================================================
# 结果展示函数:把处理结果以美观的方式渲染到页面
# ===========================================================================

def render_dish_result(result: dict, idx: int = 0):
    """
    把一道菜的检测结果渲染成 UI 卡片。

    参数:
        result: process_single_dish() 的返回值
        idx:    菜品序号(用于区分多道菜)
    """
    # 用 container 把每道菜的结果包成一个独立区块
    with st.container():
        # 标题:英文 + 繁体中文菜名
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"### 🇬🇧 {result['english']}")

        with col2:
            if result["chinese"]:
                st.markdown(f"### 🇭🇰 {result['chinese']}")
            else:
                st.markdown("### _(翻譯失敗)_")

        # 过敏原警告区
        allergens = result["allergens"]

        if not allergens:
            # 没有检测到任何过敏原
            st.success("✅ 未檢測到 14 類法定過敏原(請注意此結果僅供參考)")
        else:
            # 有过敏原:做成表格展示
            st.markdown("**⚠️ 檢測到的過敏原:**")

            # 构造表格数据
            table_data = []
            for a in allergens:
                en_label = a["label"]
                zh_label = ALLERGEN_ZH_MAP.get(en_label, en_label)
                risk_lvl = a["risk"]
                risk_zh = RISK_ZH_MAP.get(risk_lvl, risk_lvl)
                risk_em = RISK_EMOJI_MAP.get(risk_lvl, "")

                row = {
                    "過敏原 (英文)":  en_label,
                    "過敏原 (繁中)":  zh_label,
                    "風險等級":        f"{risk_em} {risk_zh}",
                    "置信度":           f"{a['score']:.2%}",
                }

                # 关键词模式下还可以显示触发的关键词
                if "hit_keywords" in a:
                    row["觸發關鍵詞"] = ", ".join(a["hit_keywords"])

                table_data.append(row)

            # 用 st.table 显示,比 st.dataframe 更美观(适合静态表格)
            st.table(table_data)

        st.divider()


# ===========================================================================
# 主界面:双 Tab 输入
# ===========================================================================

tab1, tab2 = st.tabs(["📷 上傳菜單圖片", "📝 直接輸入菜名"])


# ---------------------------------------------------------------------------
# Tab 1: 图片上传(走完整 3 个 Pipeline)
# ---------------------------------------------------------------------------

with tab1:
    st.markdown(
        """
        #### 上傳英文菜單照片
        💡 **建議**: 為獲得最佳 OCR 效果,請上傳**裁剪後的單道菜區域**圖片
        (TrOCR-small 模型主要支持單行印刷文本識別)。
        """
    )

    uploaded_file = st.file_uploader(
        "選擇圖片檔案",
        type=["jpg", "jpeg", "png", "webp"],
        help="支援 JPG / PNG / WEBP 格式,大小不超過 200MB",
    )

    if uploaded_file is not None:
        # 用 PIL 打开图片
        image = Image.open(uploaded_file)

        # 左右两列:左边显示原图,右边显示 OCR + 检测结果
        col_img, col_result = st.columns([1, 2])

        with col_img:
            st.image(image, caption="已上傳的菜單圖片", use_container_width=True)

        with col_result:
            # 提供一个按钮触发处理(避免上传就自动跑,浪费资源)
            if st.button("🚀 開始識別", type="primary", key="run_ocr_btn"):
                # Pipeline 1: OCR
                with st.spinner("Pipeline 1/3: OCR 識別中..."):
                    ocr_text = run_ocr(image)

                if not ocr_text:
                    st.error("❌ OCR 未識別到任何文字,請嘗試換一張更清晰的圖片。")
                else:
                    st.success(f"✅ OCR 結果: **{ocr_text}**")

                    # 把 OCR 输出切分成多道菜(通常只有一道,但留接口扩展)
                    dishes = split_into_dishes(ocr_text)

                    # Pipeline 2 + 3: 对每道菜跑过敏原检测和翻译
                    with st.spinner(
                        f"Pipeline 2/3: 過敏原檢測中... ({len(dishes)} 道菜)"
                    ):
                        results = [
                            process_single_dish(d, detection_mode, threshold)
                            for d in dishes
                        ]

                    # 渲染所有结果
                    st.markdown("---")
                    st.markdown("### 🔍 分析結果")
                    for i, r in enumerate(results):
                        render_dish_result(r, idx=i)


# ---------------------------------------------------------------------------
# Tab 2: 文本输入(跳过 OCR,只走 Pipeline 2 + 3)
# ---------------------------------------------------------------------------

with tab2:
    st.markdown(
        """
        #### 直接輸入英文菜名
        如果你不想上傳圖片,可以直接把菜名貼在下方。**每行一道菜**。
        """
    )

    # 提供几个示例菜名,方便用户快速测试
    example_dishes = (
        "Fish and Chips\n"
        "Grilled Salmon with Butter Sauce\n"
        "Chicken Caesar Salad\n"
        "Mushroom Risotto\n"
        "Spaghetti Carbonara"
    )

    # 让用户可以快速填入示例
    use_example = st.button("📋 載入示例菜名", key="load_example_btn")

    # 文本框,默认值取决于是否点了示例按钮
    default_text = example_dishes if use_example else ""

    menu_text = st.text_area(
        "英文菜名(每行一道)",
        value=default_text,
        height=200,
        placeholder="例如:\nFish and Chips\nGrilled Salmon\n...",
    )

    if st.button("🚀 開始分析", type="primary", key="run_text_btn"):
        # 验证输入
        if not menu_text.strip():
            st.warning("⚠️ 請先輸入至少一道菜的英文名。")
        else:
            # 按行切分菜名,过滤空行
            dishes = [
                line.strip()
                for line in menu_text.split("\n")
                if line.strip()
            ]

            # 跑 Pipeline 2 + 3
            with st.spinner(
                f"分析中... ({len(dishes)} 道菜,預計 {len(dishes) * 10} 秒)"
            ):
                results = [
                    process_single_dish(d, detection_mode, threshold)
                    for d in dishes
                ]

            # 渲染结果
            st.markdown("---")
            st.markdown(f"### 🔍 分析結果 (共 {len(results)} 道菜)")
            for i, r in enumerate(results):
                render_dish_result(r, idx=i)


# ===========================================================================
# 页面底部:再次提醒免责声明
# ===========================================================================

st.divider()
st.caption(
    "⚠️ **Disclaimer**: This tool provides AI-based allergen risk warnings "
    "and should not replace direct confirmation with restaurant staff. "
    "本工具僅供參考,實際過敏原資訊請以餐廳提供為準。"
)
