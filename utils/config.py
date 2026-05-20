"""
config.py
=========
项目核心配置:14 类英国法定过敏原标签 + 中英对照 + 关键词字典

参考来源:UK Food Standards Agency (FSA) 公布的 14 类必须标注的食物过敏原。

这个文件被 3 个 pipeline 共同依赖:
  - allergen_pipeline.py:用 ALLERGEN_LABELS 作为 zero-shot 的候选标签
  - allergen_pipeline.py:用 ALLERGEN_KEYWORDS 做关键词规则匹配
  - app.py:用 ALLERGEN_ZH_MAP 把英文结果翻译成繁体中文显示给用户
"""

# ---------------------------------------------------------------------------
# 14 类过敏原的英文标签(用于 zero-shot classification 的 candidate_labels)
# 这些标签的英文描述会被零样本模型用来与菜名/菜品描述做语义相似度匹配
# ---------------------------------------------------------------------------
ALLERGEN_LABELS = [
    "gluten",        # 麸质谷物
    "crustaceans",   # 甲壳类
    "eggs",          # 蛋类
    "fish",          # 鱼类
    "peanuts",       # 花生
    "soybeans",      # 大豆
    "milk",          # 奶类/乳制品
    "tree nuts",     # 木本坚果
    "celery",        # 芹菜
    "mustard",       # 芥末
    "sesame",        # 芝麻
    "sulphites",     # 二氧化硫及亚硫酸盐
    "lupin",         # 羽扇豆
    "molluscs",      # 软体动物
]

# ---------------------------------------------------------------------------
# 英文过敏原 → 繁体中文显示名称
# 用于最终在 UI 上向香港用户显示繁体中文提示
# ---------------------------------------------------------------------------
ALLERGEN_ZH_MAP = {
    "gluten":       "麩質穀物",
    "crustaceans":  "甲殼類",
    "eggs":         "蛋類",
    "fish":         "魚類",
    "peanuts":      "花生",
    "soybeans":     "大豆",
    "milk":         "奶類/乳製品",
    "tree nuts":    "木本堅果",
    "celery":       "芹菜",
    "mustard":      "芥末",
    "sesame":       "芝麻",
    "sulphites":    "二氧化硫及亞硫酸鹽",
    "lupin":        "羽扇豆",
    "molluscs":     "軟體動物",
}

# ---------------------------------------------------------------------------
# 关键词字典:每个过敏原对应的常见英文触发词
# 用于 Pipeline 2 的"关键词规则模式"(rule-based baseline)
# 报告里可以作为 zero-shot 模型的对比基线(baseline),凸显 deep learning 的价值
#
# 注意:这里使用小写 + 单词边界匹配,所以关键词都以小写形式出现
# ---------------------------------------------------------------------------
ALLERGEN_KEYWORDS = {
    "gluten": [
        # 含麸质谷物的常见标志词
        "wheat", "flour", "bread", "pasta", "noodle", "noodles",
        "barley", "rye", "oats", "oat", "spaghetti", "lasagna",
        "pizza", "pie", "pastry", "cake", "biscuit", "cookie",
        "cracker", "bun", "burger", "sandwich", "toast", "wrap",
        "couscous", "semolina", "dumpling", "ravioli", "gnocchi",
        "tortilla", "pancake", "waffle", "muffin", "donut", "doughnut",
        "croissant", "bagel",
    ],
    "crustaceans": [
        # 甲壳类
        "shrimp", "shrimps", "prawn", "prawns", "crab", "crabs",
        "lobster", "crayfish", "langoustine",
    ],
    "eggs": [
        # 蛋类(含蛋制品)
        "egg", "eggs", "omelette", "omelet", "mayonnaise", "mayo",
        "frittata", "quiche", "meringue", "custard", "hollandaise",
        "carbonara", "scotch egg",
    ],
    "fish": [
        # 鱼类
        "fish", "salmon", "tuna", "cod", "anchovy", "anchovies",
        "haddock", "halibut", "mackerel", "sardine", "sardines",
        "trout", "seabass", "sea bass", "snapper", "tilapia",
        "smoked salmon", "fish sauce", "kipper",
    ],
    "peanuts": [
        # 花生
        "peanut", "peanuts", "groundnut", "groundnuts", "satay",
    ],
    "soybeans": [
        # 大豆及大豆制品
        "soy", "soya", "tofu", "miso", "edamame", "tempeh",
        "soy sauce", "soya sauce", "soybean", "soya bean",
    ],
    "milk": [
        # 奶类/乳制品
        "milk", "cheese", "butter", "cream", "yogurt", "yoghurt",
        "ice cream", "whey", "casein", "lactose", "mozzarella",
        "cheddar", "parmesan", "ricotta", "feta", "brie",
        "camembert", "gouda", "custard", "ghee", "buttermilk",
        "mascarpone", "creme", "crème",
    ],
    "tree nuts": [
        # 木本坚果
        "almond", "almonds", "walnut", "walnuts", "cashew", "cashews",
        "pistachio", "pistachios", "hazelnut", "hazelnuts",
        "pecan", "pecans", "macadamia", "brazil nut", "brazil nuts",
        "pine nut", "pine nuts", "nut", "nuts",
    ],
    "celery": [
        # 芹菜
        "celery", "celeriac",
    ],
    "mustard": [
        # 芥末
        "mustard", "dijon",
    ],
    "sesame": [
        # 芝麻
        "sesame", "tahini", "sesame seed", "sesame seeds", "sesame oil",
    ],
    "sulphites": [
        # 二氧化硫及亚硫酸盐(常见于葡萄酒、果干等)
        "wine", "champagne", "prosecco", "sherry", "dried fruit",
        "raisin", "raisins", "sultana", "sultanas", "apricot",
        "balsamic",
    ],
    "lupin": [
        # 羽扇豆(欧洲烘焙食品中较常见)
        "lupin", "lupini", "lupine",
    ],
    "molluscs": [
        # 软体动物
        "oyster", "oysters", "mussel", "mussels", "clam", "clams",
        "squid", "calamari", "octopus", "scallop", "scallops",
        "snail", "escargot",
    ],
}

# ---------------------------------------------------------------------------
# 风险等级阈值(基于过敏原置信度得分)
# 用于把模型输出的概率值转换成给用户看的 High / Medium / Low 风险等级
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "high":    0.75,   # 置信度 >= 0.75 视为高风险
    "medium":  0.50,   # 0.50 - 0.75 中等风险
    "low":     0.30,   # 0.30 - 0.50 低风险
    # < 0.30 不报警
}

# 风险等级 → 繁体中文显示
RISK_ZH_MAP = {
    "high":   "高風險",
    "medium": "中度風險",
    "low":    "低風險",
    "none":   "未檢測到",
}

# 风险等级 → emoji 颜色提示(UI 显示用)
RISK_EMOJI_MAP = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
    "none":   "⚪",
}


def get_risk_level(score: float) -> str:
    """
    根据置信度得分返回风险等级字符串。

    参数:
        score: 模型输出的过敏原置信度,范围 [0, 1]

    返回:
        "high" / "medium" / "low" / "none" 其中之一
    """
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    elif score >= RISK_THRESHOLDS["low"]:
        return "low"
    else:
        return "none"
