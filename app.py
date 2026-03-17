import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="石璞會館 AI 評論分析", layout="wide")

# =========================
# 1. 指標規則（正向 / 負向）
# =========================
INDICATORS = [
    "地點便利",
    "服務熱情",
    "房間整潔",
    "睡眠品質",
    "設施環境",
    "交通便利",
    "在地體驗",
    "CP值"
]

INDICATOR_RULES = {
    "地點便利": {
        "positive": ["地點方便", "方便", "附近", "位置好", "鬧區", "景點", "近", "美食街"],
        "negative": ["偏僻", "難找", "太遠", "不方便"]
    },
    "服務熱情": {
        "positive": ["老闆熱情", "闆娘熱情", "熱情", "親切", "友善", "幫忙", "貼心", "服務好", "接待好"],
        "negative": ["冷淡", "態度差", "不耐煩", "愛理不理", "服務差"]
    },
    "房間整潔": {
        "positive": ["乾淨", "整潔", "舒適", "清潔", "整齊"],
        "negative": ["髒", "不乾淨", "灰塵", "凌亂", "發霉"]
    },
    "睡眠品質": {
        "positive": ["好睡", "安靜", "床舒服", "枕頭舒服", "睡得很好"],
        "negative": ["隔音差", "噪音", "吵", "太吵", "不好睡"]
    },
    "設施環境": {
        "positive": ["陽台", "露台", "設備齊全", "裝潢漂亮", "環境好", "空間舒適", "浴室乾淨"],
        "negative": ["設備老舊", "潮濕", "浴室台階高", "空間小", "設備差"]
    },
    "交通便利": {
        "positive": ["停車方便", "交通方便", "機場近", "方便停車", "租車方便"],
        "negative": ["停車不便", "交通不便", "難停車"]
    },
    "在地體驗": {
        "positive": ["風獅爺", "伴手禮", "金門特色", "導覽", "老街", "在地體驗"],
        "negative": []
    },
    "CP值": {
        "positive": ["值得", "划算", "便宜", "超值", "cp值高", "價格合理"],
        "negative": ["太貴", "不值得", "cp值低", "價格偏高"]
    }
}

PROBLEM_KEYWORDS = ["隔音差", "噪音", "吵", "台階高", "浴室台階高", "潮濕", "蚊子", "小", "舊", "不方便"]


BUSINESS_IMPORTANCE = {
    "地點便利": 0.7,
    "服務熱情": 0.9,
    "房間整潔": 1.0,
    "睡眠品質": 0.9,
    "設施環境": 0.7,
    "交通便利": 0.6,
    "在地體驗": 0.5,
    "CP值": 0.6,
}

ISSUE_RULES = {
    "噪音問題": ["隔音差", "噪音", "吵", "太吵"],
    "浴室問題": ["浴室台階高", "台階高", "浴室較舊", "浴室潮濕"],
    "設備問題": ["設備老舊", "設備差", "冷氣不好", "設備故障"],
    "蚊蟲問題": ["蚊子", "蟲", "蚊蟲"],
    "空間問題": ["空間小", "房間小", "太小"]
}

def calculate_market_customer_weights(all_scores):
    market_mentions = all_scores.groupby("indicator")["mention_count"].sum().reset_index()
    total_mentions = market_mentions["mention_count"].sum()

    if total_mentions == 0:
        market_mentions["customer_W"] = 0
    else:
        market_mentions["customer_W"] = market_mentions["mention_count"] / total_mentions

    market_mentions["customer_W"] = market_mentions["customer_W"].round(4)
    return market_mentions[["indicator", "customer_W"]]

VALUE_ALPHA = 1.0
VALUE_BETA = 0.7

# =========================
# 2. 載入資料
# =========================
@st.cache_data
def load_data():
    products = pd.read_csv("data/products.csv")
    reviews = pd.read_csv("data/reviews.csv", on_bad_lines="skip")
    return products, reviews

# =========================
# 3. 分析函數
# =========================

def render_hero_section(title, subtitle, own_name, own_price, total_reviews):
    html = f"""
<div style="background: linear-gradient(135deg, #0F2A44 0%, #2F80ED 100%);
            padding: 28px 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);">
    <div style="font-size: 14px; opacity: 0.85; margin-bottom: 8px;">
        AI Market Positioning Prototype
    </div>
    <div style="font-size: 34px; font-weight: 700; margin-bottom: 10px;">
        {title}
    </div>
    <div style="font-size: 17px; line-height: 1.6; opacity: 0.95; margin-bottom: 18px;">
        {subtitle}
    </div>
    <div style="display: flex; gap: 18px; flex-wrap: wrap; font-size: 15px;">
        <div><b>分析對象：</b>{own_name}</div>
        <div><b>平均價格：</b>{own_price}</div>
        <div><b>評論總數：</b>{total_reviews}</div>
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_info_card(title, content, border_color="#2F80ED", bg_color="#F8FAFC"):
    clean_content = content.strip().replace("\n", "<br>")

    html = f"""
<div style="background: {bg_color};
            border-left: 6px solid {border_color};
            padding: 18px 20px;
            border-radius: 14px;
            margin-bottom: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
    <div style="font-size: 18px;
                font-weight: 700;
                color: #0F2A44;
                margin-bottom: 8px;">
        {title}
    </div>
    <div style="font-size: 15px;
                color: #334155;
                line-height: 1.8;">
        {clean_content}
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

def render_kpi_card(label, value, help_text="", color="#2F80ED"):
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #E5E7EB;
        border-top: 5px solid {color};
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        min-height: 120px;
    ">
        <div style="
            font-size: 13px;
            color: #64748B;
            margin-bottom: 10px;
        ">
            {label}
        </div>
        <div style="
            font-size: 30px;
            font-weight: 700;
            color: #0F2A44;
            margin-bottom: 8px;
        ">
            {value}
        </div>
        <div style="
            font-size: 13px;
            color: #94A3B8;
            line-height: 1.5;
        ">
            {help_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_highlight_block(title, items, kind="success"):
    st.subheader(title)

    if not items:
        if kind == "success":
            st.info("目前沒有可顯示的項目")
        else:
            st.info("目前沒有明顯修正項")
        return

    for item in items:
        box_text = f"**{item['title']}**\n\n{item['desc']}"
        if kind == "success":
            st.success(box_text)
        elif kind == "warning":
            st.warning(box_text)
        else:
            st.info(box_text)


def build_strength_items(df, score_col, top_n=3, mode="quality"):
    items = []

    if df.empty:
        return items

    top_df = df.sort_values(score_col, ascending=False).head(top_n)

    for _, row in top_df.iterrows():
        indicator = row["indicator"]

        if mode == "quality":
            desc = f"在同價格帶中，此指標顯著優於市場平均（Z = {row['Z']:.2f}），具備差異化競爭優勢，可作為主要賣點進行放大。"

        elif mode == "value_fix":
            desc = f"此指標的價值表現低於市場預期（value_z = {row['value_z']:.2f}），代表價格與體驗存在落差，建議優先進行優化。"

        items.append({
            "title": indicator,
            "desc": desc
        })

    return items
def calculate_market_stats(all_scores: pd.DataFrame, competitor_products: pd.DataFrame) -> pd.DataFrame:
    """
    根據競品分數表計算每個 indicator 的市場平均、標準差、競品數
    """
    competitor_ids = competitor_products["product_id"].tolist()

    competitor_scores = all_scores[
        all_scores["product_id"].isin(competitor_ids)
    ].copy()

    market_stats = (
        competitor_scores.groupby("indicator", as_index=False)
        .agg(
            mu=("adjusted_score", "mean"),
            sigma=("adjusted_score", "std"),
            competitor_n=("product_id", "nunique")
        )
    )

    market_stats["sigma"] = market_stats["sigma"].fillna(0)

    return market_stats

def assign_price_segment(products: pd.DataFrame, q: int = 3) -> pd.DataFrame:
    """
    依價格分位數切市場區隔
    q=3 -> low / mid / high
    """
    products = products.copy()

    labels = ["low", "mid", "high"]

    # 避免價格重複太多時 qcut 出錯
    products["price_segment"] = pd.qcut(
        products["price"],
        q=q,
        labels=labels,
        duplicates="drop"
    )

    products["price_segment"] = products["price_segment"].astype(str)

    return products


def get_own_segment(products: pd.DataFrame, own_id: str) -> str:
    """
    取得 OWN 所在的價格區隔
    """
    own_segment = products.loc[
        products["product_id"] == own_id, "price_segment"
    ].iloc[0]

    return own_segment


def filter_segment_products(products: pd.DataFrame, own_segment: str) -> pd.DataFrame:
    """
    只保留同價格區隔的產品
    """
    return products[products["price_segment"] == own_segment].copy()


def filter_segment_scores(all_scores: pd.DataFrame, segment_products: pd.DataFrame) -> pd.DataFrame:
    """
    只保留同價格區隔產品的分數表
    """
    valid_ids = segment_products["product_id"].tolist()
    return all_scores[all_scores["product_id"].isin(valid_ids)].copy()

def calculate_market_reliability(all_scores, products, range_threshold=3.0, sigma_threshold=2.0, min_competitors=5):
    competitor_ids = products.loc[products["is_own"] == 0, "product_id"]
    competitor_scores = all_scores[all_scores["product_id"].isin(competitor_ids)].copy()

    rows = []

    for indicator in competitor_scores["indicator"].unique():
        subset = competitor_scores[competitor_scores["indicator"] == indicator]

        competitor_n = subset["product_id"].nunique()
        max_score = subset["adjusted_score"].max()
        min_score = subset["adjusted_score"].min()
        score_range = max_score - min_score
        sigma = subset["adjusted_score"].std()

        if pd.isna(sigma):
            sigma = 0.0

        low_flag = (
            competitor_n < min_competitors
            or score_range > range_threshold
            or sigma > sigma_threshold
        )

        rows.append({
            "indicator": indicator,
            "competitor_n": competitor_n,
            "score_range": round(score_range, 4),
            "market_sigma_check": round(sigma, 4),
            "market_reliability": "low" if low_flag else "normal"
        })

    return pd.DataFrame(rows)

def normalize_business_weights(business_importance):
    total = sum(business_importance.values())

    if total == 0:
        return {k: 0 for k in business_importance.keys()}

    normalized = {
        k: v / total
        for k, v in business_importance.items()
    }

    return normalized

def count_sentiment_hits_by_indicator(reviews_df, indicator_rules):
    rows = []

    for indicator, rules in indicator_rules.items():
        pos_keywords = rules.get("positive", [])
        neg_keywords = rules.get("negative", [])

        pos_count = 0
        neg_count = 0

        for text in reviews_df["review_text"].fillna("").astype(str):
            if any(kw in text for kw in pos_keywords):
                pos_count += 1
            if any(kw in text for kw in neg_keywords):
                neg_count += 1

        rows.append({
            "indicator": indicator,
            "positive_count": pos_count,
            "negative_count": neg_count
        })

    result = pd.DataFrame(rows)
    return result


def calculate_net_scores(df_counts, k=5):
    result = df_counts.copy()

    result["mention_count"] = result["positive_count"] + result["negative_count"]

    result["raw_score"] = (
        (result["positive_count"] - result["negative_count"]) /
        (result["positive_count"] + result["negative_count"] + 1)
    )

    result["score"] = 5 + 5 * result["raw_score"]
    result["score"] = result["score"].clip(lower=0, upper=10)

    # 反向指標修正：問題點越低越好，所以要反轉
    reverse_indicators = ["問題點"]
    result.loc[result["indicator"].isin(reverse_indicators), "score"] = (
        10 - result.loc[result["indicator"].isin(reverse_indicators), "score"]
    )

    # 小樣本修正
    result["confidence_factor"] = result["mention_count"] / (result["mention_count"] + k)
    result["adjusted_score"] = 5 + (result["score"] - 5) * result["confidence_factor"]
    result["adjusted_score"] = result["adjusted_score"].round(2)

    # score 也保留兩位
    result["score"] = result["score"].round(2)
    result["raw_score"] = result["raw_score"].round(4)
    result["confidence_factor"] = result["confidence_factor"].round(4)

    return result


def build_radar_chart(score_df):
    labels = score_df["indicator"].tolist()
    values = score_df["score"].tolist()

    labels = labels + [labels[0]]
    values = values + [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    ax.set_ylim(0, 10)

    return fig


def build_segment_sentiment_table(reviews_df, segment_col, indicator_rules):
    segment_values = reviews_df[segment_col].fillna("未知").astype(str).unique().tolist()
    rows = []

    for seg in segment_values:
        subset = reviews_df[reviews_df[segment_col].fillna("未知").astype(str) == seg]
        row = {segment_col: seg}

        for indicator, rules in indicator_rules.items():
            pos_keywords = rules.get("positive", [])
            neg_keywords = rules.get("negative", [])

            pos_count = 0
            neg_count = 0

            for text in subset["review_text"].fillna("").astype(str):
                if any(kw in text for kw in pos_keywords):
                    pos_count += 1
                if any(kw in text for kw in neg_keywords):
                    neg_count += 1

            row[f"{indicator}_正向"] = pos_count
            row[f"{indicator}_負向"] = neg_count

        rows.append(row)

    return pd.DataFrame(rows)


def build_problem_priority_table(reviews_df, problem_keywords):
    rows = []

    for kw in problem_keywords:
        mask = reviews_df["review_text"].fillna("").astype(str).str.contains(kw, na=False)
        subset = reviews_df[mask].copy()

        mention_count = len(subset)

        if "companion_type" in subset.columns:
            affected_segments = subset["companion_type"].fillna("未知").nunique()
        else:
            affected_segments = 1 if mention_count > 0 else 0

        priority_score = mention_count * affected_segments

        rows.append({
            "problem_keyword": kw,
            "mention_count": mention_count,
            "affected_segments": affected_segments,
            "priority_score": priority_score
        })

    result = pd.DataFrame(rows).sort_values(
        ["priority_score", "mention_count"], ascending=False
    )
    return result


def get_positive_negative_evidence(reviews_df, keyword, max_rows=8):
    mask = reviews_df["review_text"].fillna("").astype(str).str.contains(keyword, na=False)
    matched = reviews_df.loc[mask].copy()

    positive_hints = ["乾淨", "舒適", "方便", "熱情", "親切", "友善", "值得", "推薦", "好睡", "漂亮"]
    negative_hints = ["隔音差", "噪音", "吵", "髒", "潮濕", "小", "舊", "不方便", "台階高", "蚊子"]

    def classify_text(text):
        text = str(text)
        pos_hit = any(w in text for w in positive_hints)
        neg_hit = any(w in text for w in negative_hints)

        if pos_hit and not neg_hit:
            return "正面"
        elif neg_hit and not pos_hit:
            return "負面"
        elif pos_hit and neg_hit:
            return "混合"
        else:
            return "中性"

    matched["evidence_type"] = matched["review_text"].apply(classify_text)

    cols = ["companion_type", "nationality", "stay_days", "pros", "cons", "review_text", "evidence_type"]
    existing_cols = [c for c in cols if c in matched.columns]
    matched = matched[existing_cols]

    pos_df = matched[matched["evidence_type"] == "正面"].head(max_rows)
    neg_df = matched[matched["evidence_type"] == "負面"].head(max_rows)
    mix_df = matched[matched["evidence_type"] == "混合"].head(max_rows)

    return pos_df, neg_df, mix_df


def generate_marketing_suggestions(indicator_scores):
    top3 = indicator_scores.sort_values("score", ascending=False).head(3)["indicator"].tolist()
    suggestions = []

    mapping = {
        "地點便利": "可主打『交通方便、景點接近、第一次來金門也容易安排』。",
        "服務熱情": "可主打『老闆 / 闆娘親切熱情、人情味強、住宿體驗溫暖』。",
        "房間整潔": "可主打『房間乾淨整潔、住起來安心舒適』。",
        "睡眠品質": "可主打『安靜、好睡、適合放鬆休息』。",
        "設施環境": "可主打『環境舒適、設施齊全、空間體驗佳』。",
        "交通便利": "可主打『停車方便、移動便利、適合自駕客』。",
        "在地體驗": "可主打『金門特色、在地感、旅遊體驗完整』。",
        "CP值": "可主打『價格合理、性價比高、花費與體驗相符』。",
        "問題點": "此構面不適合作為行銷主軸，建議優先改善。"
    }

    for ind in top3:
        suggestions.append({
            "優勢構面": ind,
            "建議文案方向": mapping.get(ind, "可作為住宿特色進一步包裝。")
        })

    return pd.DataFrame(suggestions)

def calculate_market_benchmark(all_scores, products):
    competitor_ids = products.loc[products["is_own"] == 0, "product_id"]
    competitor_scores = all_scores[all_scores["product_id"].isin(competitor_ids)]

    market_stats = competitor_scores.groupby("indicator")["adjusted_score"].agg(
        mu="mean",
        sigma="std"
    ).reset_index()

    market_stats["sigma"] = market_stats["sigma"].replace(0, 0.1)
    market_stats["sigma"] = market_stats["sigma"].fillna(0.1)

    return market_stats

def calculate_z_scores(own_scores, market_stats):

    merged = own_scores.merge(
        market_stats,
        on="indicator",
        how="left"
    )

    merged["Z"] = (merged["score"] - merged["mu"]) / merged["sigma"]

    return merged

def calculate_weights(score_table, market_customer_w, business_importance, alpha=0.7, beta=0.3):
    result = score_table.copy()

    # 先把 business importance 正規化
    normalized_business_w = normalize_business_weights(business_importance)

    # merge 市場 customer weight
    result = result.merge(market_customer_w, on="indicator", how="left")

    # map 正規化後的 business weight
    result["business_W"] = result["indicator"].map(normalized_business_w)

    result["customer_W"] = result["customer_W"].fillna(0)
    result["business_W"] = result["business_W"].fillna(0)

    # final fused weight
    result["W"] = alpha * result["customer_W"] + beta * result["business_W"]

    result["customer_W"] = result["customer_W"].round(4)
    result["business_W"] = result["business_W"].round(4)
    result["W"] = result["W"].round(4)

    return result

def calculate_action_priority(df):
    result = df.copy()

    # 品質放大：品質本身很強
    result["quality_amplify_priority"] = (
        result["Z"].apply(lambda x: max(0, x))
        * result["W"]
        * result["confidence_factor"]
    )

    # 價值放大：考慮價格後仍然很有競爭力
    result["value_amplify_priority"] = (
        result["value_z"].apply(lambda x: max(0, x))
        * result["W"]
        * result["confidence_factor"]
    )

    # 品質修復：品質本身落後市場
    result["quality_fix_priority"] = (
        result["Z"].apply(lambda x: max(0, -x))
        * result["W"]
        * result["confidence_factor"]
    )

    # 價值修復：價值不足（品質與價格不匹配）
    result["value_fix_priority"] = (
        result["value_z"].apply(lambda x: max(0, -x))
        * result["W"]
        * result["confidence_factor"]
    )

    for col in [
        "quality_amplify_priority",
        "value_amplify_priority",
        "quality_fix_priority",
        "value_fix_priority"
    ]:
        result[col] = result[col].round(4)

    return result

SIGMA_FLOOR = 0.8
OWN_MENTION_MIN = 5
MARKET_MENTION_MIN = 5
COMPETITOR_MIN = 5


def add_market_mention_stats(all_scores: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個 indicator 在競品市場中的平均 mention_count
    """
    competitor_scores = all_scores[all_scores["product_id"] != "OWN_001"].copy()

    market_mention = (
        competitor_scores.groupby("indicator", as_index=False)["mention_count"]
        .mean()
        .rename(columns={"mention_count": "market_avg_mention"})
    )

    comparison = comparison.merge(market_mention, on="indicator", how="left")
    comparison["market_avg_mention"] = comparison["market_avg_mention"].fillna(0)

    return comparison


def add_signal_flags(comparison: pd.DataFrame,
                     own_mention_min: int = OWN_MENTION_MIN,
                     market_mention_min: int = MARKET_MENTION_MIN) -> pd.DataFrame:
    """
    低訊號指標判斷
    """
    comparison["low_signal_indicator"] = np.where(
        (comparison["mention_count"] < own_mention_min) |
        (comparison["market_avg_mention"] < market_mention_min),
        1,
        0
    )
    return comparison


def apply_sigma_floor(comparison: pd.DataFrame,
                      sigma_floor: float = SIGMA_FLOOR) -> pd.DataFrame:
    """
    套用 sigma floor，避免 sigma 太小導致 Z 爆炸
    """
    comparison["sigma_raw"] = comparison["sigma"].copy()
    comparison["sigma_used"] = comparison["sigma"].clip(lower=sigma_floor)

    comparison["Z"] = (
        (comparison["adjusted_score"] - comparison["mu"]) / comparison["sigma_used"]
    )

    if "price_z" in comparison.columns:
        comparison["value_z"] = comparison["Z"] - comparison["price_z"]

    return comparison


def classify_indicator_reliability(row,
                                   own_mention_min: int = OWN_MENTION_MIN,
                                   market_mention_min: int = MARKET_MENTION_MIN,
                                   sigma_floor: float = SIGMA_FLOOR,
                                   competitor_min: int = COMPETITOR_MIN) -> str:
    """
    單一指標層級可靠度
    """
    if row["mention_count"] < own_mention_min:
        return "low"

    if row["market_avg_mention"] < market_mention_min:
        return "low"

    if row["sigma_raw"] < sigma_floor:
        return "low"

    if row["competitor_n"] < competitor_min:
        return "low"

    return "normal"

def classify_quality_strategy(row):
    if row["low_signal_indicator"] == 1:
        return "低訊號指標"

    z = row["Z"]

    if z >= 2:
        return "核心優勢"
    elif z >= 1:
        return "優勢"
    elif z > -1:
        return "市場平均"
    elif z > -2:
        return "弱點"
    else:
        return "核心弱點"


def classify_value_strategy(row):
    if row["low_signal_indicator"] == 1:
        return "低訊號指標"

    v = row["value_z"]

    if v >= 2:
        return "核心優勢"
    elif v >= 1:
        return "優勢"
    elif v > -1:
        return "市場平均"
    elif v > -2:
        return "弱點"
    else:
        return "核心弱點"

def calculate_price_benchmark(products):
    competitor_prices = products.loc[products["is_own"] == 0, "price"].astype(float)

    price_mu = competitor_prices.mean()
    price_sigma = competitor_prices.std()

    if pd.isna(price_sigma) or price_sigma == 0:
        price_sigma = 0.1

    return price_mu, price_sigma


def calculate_value_scores(comparison_df, own_price, price_mu, price_sigma, alpha=1.0, beta=0.7):
    result = comparison_df.copy()

    result["price_z"] = (own_price - price_mu) / price_sigma
    result["value_z"] = alpha * result["Z"] - beta * result["price_z"]

    result["price_z"] = result["price_z"].round(4)
    result["value_z"] = result["value_z"].round(4)

    return result


# =========================
# 4. 主畫面
# =========================


st.title("石璞會館 AI 評論分析工具 v4")
dev_mode = False
products, reviews = load_data()

# ===== 價格分段 =====
own_id = "OWN_001"

products = assign_price_segment(products, q=3)
own_segment = get_own_segment(products, own_id)
segment_products = filter_segment_products(products, own_segment)
segment_competitors = segment_products[segment_products["is_own"] == 0].copy()
st.success("資料讀取成功")

own_name = products.loc[products["product_id"] == own_id, "name"].iloc[0]
own_price_display = int(products.loc[products["product_id"] == own_id, "price"].iloc[0])
total_reviews_display = len(reviews[reviews["product_id"] == own_id])

render_hero_section(
    title="金門民宿市場定位分析系統",
    subtitle="以價格區隔市場，分析同級競品中的品質定位、價值定位與改善方向。",
    own_name=own_name,
    own_price=f"{own_price_display}",
    total_reviews=total_reviews_display
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi_card(
        "平均價格",
        f"{own_price_display}",
        "OWN 平均價格",
        color="#2F80ED"
    )

with col2:
    render_kpi_card(
        "評論總數",
        f"{total_reviews_display}",
        "目前已納入分析的 OWN 評論數",
        color="#2D9CDB"
    )

with col3:
    render_kpi_card(
        "市場區隔",
        f"{own_segment}",
        "依價格分位數切出的 segment",
        color="#27AE60"
    )

with col4:
    render_kpi_card(
        "同級競品數",
        f"{len(segment_competitors)}",
        "同價格區隔的競爭對手數量",
        color="#C9A227"
    )

render_info_card(
    "市場定位摘要",
    f"""
    目前 {own_name} 被歸類在 <b>{own_segment}</b> 價格區隔中，
    並與同區隔的 <b>{len(segment_competitors)}</b> 家競品進行比較。
    模型會先計算各項品質指標的 adjusted score，
    再與同級市場 benchmark 比較，產出 Z、value_z 與策略建議。
    """,
    border_color="#2F80ED",
    bg_color="#F8FAFC"
)

render_info_card(
    "系統價值摘要",
    f"{own_name} 的分析結果顯示，本系統可透過價格區隔與同級市場比較，快速定位品質優勢與價值落差，並輸出可執行的優化策略。此模型可應用於民宿、餐飲與零售產業，作為數據驅動的經營決策工具。",
    border_color="#27AE60",
    bg_color="#F6FFF8"
)   

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "九項品質",
    "客群分析",
    "改善優先",
    "證據分析",
    "原始資料",
    "市場比較"
])

# =========================
# Tab 1 九項品質
# =========================


with tab1:
    st.subheader("九項品質指標（正負向淨分數）")

    sentiment_counts = count_sentiment_hits_by_indicator(reviews, INDICATOR_RULES)
    indicator_scores = calculate_net_scores(sentiment_counts)

    left, right = st.columns([1.2, 1])

    with left:
        st.dataframe(indicator_scores, use_container_width=True)

    with right:
        fig = build_radar_chart(indicator_scores[["indicator", "score"]])
        st.pyplot(fig)

    st.markdown("### 綜合摘要")
    best_indicator = indicator_scores.sort_values("score", ascending=False).iloc[0]
    worst_indicator = indicator_scores.sort_values("score", ascending=True).iloc[0]

    st.write(f"目前最強的構面：**{best_indicator['indicator']}**（分數 {best_indicator['score']}）")
    st.write(f"目前最弱的構面：**{worst_indicator['indicator']}**（分數 {worst_indicator['score']}）")

    st.markdown("### 優勢行銷建議")
    marketing_df = generate_marketing_suggestions(indicator_scores)
    st.dataframe(marketing_df, use_container_width=True)

# =========================
# Tab 2 客群分析
# =========================
with tab2:
    st.subheader("客群交叉分析（正向 / 負向）")

    st.markdown("### 同行類型 × 指標正負向計數")
    if "companion_type" in reviews.columns:
        segment_companion = build_segment_sentiment_table(reviews, "companion_type", INDICATOR_RULES)
        st.dataframe(segment_companion, use_container_width=True)
    else:
        st.warning("reviews.csv 缺少 companion_type 欄位")

    st.markdown("### 國籍 × 指標正負向計數")
    if "nationality" in reviews.columns:
        segment_nationality = build_segment_sentiment_table(reviews, "nationality", INDICATOR_RULES)
        st.dataframe(segment_nationality, use_container_width=True)
    else:
        st.warning("reviews.csv 缺少 nationality 欄位")

# =========================
# Tab 3 改善優先
# =========================
with tab3:
    st.subheader("改善優先排序（負向問題）")

    priority_df = build_problem_priority_table(reviews, PROBLEM_KEYWORDS)
    st.dataframe(priority_df, use_container_width=True)

    top3_fix = priority_df.head(3)

    st.markdown("### Top 3 改善建議摘要")
    for _, row in top3_fix.iterrows():
        st.write(
            f"- **{row['problem_keyword']}**：提及 {int(row['mention_count'])} 次，"
            f"影響 {int(row['affected_segments'])} 類客群，優先度 {int(row['priority_score'])}"
        )

# =========================
# Tab 4 證據分析
# =========================
with tab4:
    st.subheader("證據分析（正面 / 負面 / 混合）")

    keyword = st.text_input("輸入想查看的關鍵字，例如：隔音 / 乾淨 / 熱情 / 地點", value="隔音")

    if keyword:
        pos_df, neg_df, mix_df = get_positive_negative_evidence(reviews, keyword)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### 正面證據")
            st.dataframe(pos_df, use_container_width=True)

        with c2:
            st.markdown("### 負面證據")
            st.dataframe(neg_df, use_container_width=True)

        with c3:
            st.markdown("### 混合證據")
            st.dataframe(mix_df, use_container_width=True)

# =========================
# Tab 5 原始資料
# =========================
with tab5:
    st.subheader("原始資料")
    with st.expander("產品資料"):
        st.dataframe(products, use_container_width=True)

    with st.expander("評論資料（前30筆）"):
        st.dataframe(reviews.head(30), use_container_width=True)

# =========================
# Tab 6 原始資料
# =========================
with tab6:

    all_scores = []

    for pid in products["product_id"].unique():
        subset = reviews[reviews["product_id"] == pid]

        sentiment_counts = count_sentiment_hits_by_indicator(
            subset,
            INDICATOR_RULES
        )

        scores = calculate_net_scores(sentiment_counts)
        scores["product_id"] = pid
        all_scores.append(scores)

    all_scores = pd.concat(all_scores, ignore_index=True)

# ===== 只保留同價格區隔的分數（一定要放在 concat 後）=====
segment_scores = filter_segment_scores(all_scores, segment_products)

# 這裡開始全部改成 segment benchmark
market_stats = calculate_market_stats(segment_scores, segment_competitors)
market_customer_w = calculate_market_customer_weights(segment_scores)
market_reliability_df = calculate_market_reliability(segment_scores, segment_products)

own_id = products.loc[products["is_own"] == 1, "product_id"].iloc[0]
own_scores = segment_scores[segment_scores["product_id"] == own_id].copy()

# ===== 基本 comparison =====
comparison = calculate_z_scores(own_scores, market_stats)

price_mu, price_sigma = calculate_price_benchmark(segment_products)
own_price = float(
    segment_products.loc[segment_products["product_id"] == own_id, "price"].iloc[0]
)

comparison = calculate_value_scores(
    comparison,
    own_price,
    price_mu,
    price_sigma,
    alpha=VALUE_ALPHA,
    beta=VALUE_BETA
)

comparison = calculate_weights(
    comparison,
    market_customer_w,
    BUSINESS_IMPORTANCE
)

# ===== 這裡再補 market_stats 欄位，避免被前面函數蓋掉 =====
comparison = comparison.merge(
    market_stats[["indicator", "mu", "sigma", "competitor_n"]],
    on="indicator",
    how="left",
    suffixes=("", "_market")
)

# 如果前面函數已經有 mu / sigma，保留原本；若沒有就補 market_stats 的
if "mu_market" in comparison.columns:
    if "mu" in comparison.columns:
        comparison["mu"] = comparison["mu"].fillna(comparison["mu_market"])
    else:
        comparison["mu"] = comparison["mu_market"]

if "sigma_market" in comparison.columns:
    if "sigma" in comparison.columns:
        comparison["sigma"] = comparison["sigma"].fillna(comparison["sigma_market"])
    else:
        comparison["sigma"] = comparison["sigma_market"]

drop_cols = [c for c in ["mu_market", "sigma_market"] if c in comparison.columns]
if drop_cols:
    comparison = comparison.drop(columns=drop_cols)

# merge 市場可靠度旗標
comparison = comparison.merge(
    market_reliability_df,
    on="indicator",
    how="left"
)

# ===== 先做訊號治理 =====

# 1. 加入市場平均 mention
comparison = add_market_mention_stats(segment_scores, comparison)

# 2. 加入 low signal flag
comparison = add_signal_flags(comparison)

# 3. 套用 sigma floor，重算 Z 與 value_z
comparison = apply_sigma_floor(comparison)

# ===== 統一 competitor_n 欄位 =====
competitor_candidates = [
    "competitor_n",
    "competitor_n_x",
    "competitor_n_market",
    "competitor_n_y"
]

existing_competitor_cols = [c for c in competitor_candidates if c in comparison.columns]

if "competitor_n" not in comparison.columns:
    if existing_competitor_cols:
        comparison["competitor_n"] = comparison[existing_competitor_cols].bfill(axis=1).iloc[:, 0]
    else:
        comparison["competitor_n"] = np.nan

# 清理重複欄位，只保留正式的 competitor_n
drop_competitor_cols = [
    c for c in ["competitor_n_x", "competitor_n_market", "competitor_n_y"]
    if c in comparison.columns
]
if drop_competitor_cols:
    comparison = comparison.drop(columns=drop_competitor_cols)

# 4. 防呆：先確認 indicator_reliability 需要的欄位都在
required_before_reliability = [
    "indicator",
    "mention_count",
    "market_avg_mention",
    "sigma",
    "sigma_raw",
    "sigma_used",
    "competitor_n"
]

missing_before_reliability = [
    c for c in required_before_reliability if c not in comparison.columns
]

if missing_before_reliability:
    st.error(f"indicator_reliability 前缺少欄位: {missing_before_reliability}")
    st.write("目前 comparison 欄位：", comparison.columns.tolist())
    st.stop()

# 5. 單一指標可靠度
comparison["indicator_reliability"] = comparison.apply(
    classify_indicator_reliability,
    axis=1
)

# 6. 防呆檢查
required_cols = [
    "market_avg_mention",
    "low_signal_indicator",
    "sigma_raw",
    "sigma_used",
    "indicator_reliability",
    "competitor_n"
]

missing_cols = [c for c in required_cols if c not in comparison.columns]
if missing_cols:
    st.error(f"comparison 缺少欄位: {missing_cols}")
    st.write("目前 comparison 欄位：", comparison.columns.tolist())
    st.stop()
if dev_mode:
    st.subheader("DEBUG: segment info")
    st.write("OWN segment:", own_segment)
    st.write("Segment product count:", len(segment_products))
    st.write("Segment competitor count:", len(segment_competitors))
    st.dataframe(
        segment_products[["product_id", "name", "price", "price_segment", "is_own"]],
        use_container_width=True
    )

if dev_mode:
    st.write("DEBUG comparison columns:", comparison.columns.tolist())

# ===== 再做 strategy =====
comparison["quality_strategy"] = comparison.apply(classify_quality_strategy, axis=1)
comparison["value_strategy"] = comparison.apply(classify_value_strategy, axis=1)

# ===== 再做 priority =====
comparison["quality_amplify_priority"] = np.where(
    comparison["low_signal_indicator"] == 1,
    0,
    np.maximum(comparison["Z"], 0) * comparison["W"]
)

comparison["value_amplify_priority"] = np.where(
    comparison["low_signal_indicator"] == 1,
    0,
    np.maximum(comparison["value_z"], 0) * comparison["W"]
)

comparison["quality_fix_priority"] = np.where(
    comparison["low_signal_indicator"] == 1,
    0,
    np.maximum(-comparison["Z"], 0) * comparison["W"]
)

comparison["value_fix_priority"] = np.where(
    comparison["low_signal_indicator"] == 1,
    0,
    np.maximum(-comparison["value_z"], 0) * comparison["W"]
)

st.subheader("市場比較（Benchmark）")

render_info_card(
    "市場定位摘要",
    f"{own_name} 目前位於 <b>{own_segment}</b> 價格區隔，並與同價格帶的 <b>{len(segment_competitors)}</b> 家競品比較。"
    f"目前模型使用同級市場 benchmark 計算各指標的品質定位 Z 與價值定位 value_z。",
    border_color="#2F80ED",
    bg_color="#F8FAFC"
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    render_kpi_card(
        "價格區隔",
        f"{own_segment}",
        "目前 OWN 所屬的 segment",
        color="#2F80ED"
    )

with col_b:
    render_kpi_card(
        "同級競品數",
        f"{len(segment_competitors)}",
        "同價格帶比較對象",
        color="#27AE60"
    )

with col_c:
    render_kpi_card(
        "價格定位",
        f"{comparison['price_z'].iloc[0]:.2f}" if len(comparison) > 0 else "0.00",
        "0 附近代表接近同級均價",
        color="#C9A227"
    )

st.markdown("### 核心輸出")

left_col, right_col = st.columns(2)

with left_col:
    quality_candidates = comparison[
        (comparison["low_signal_indicator"] == 0) &
        (comparison["quality_amplify_priority"] > 0)
    ].copy()

    quality_items = build_strength_items(
        quality_candidates,
        "quality_amplify_priority",
        top_n=3,
        mode="quality"
    )

    render_highlight_block(
    "品質優勢建議",
    quality_items,
    kind="success"
    )

with right_col:
    value_fix_candidates = comparison.sort_values("value_z").head(3).copy()

    value_fix_items = build_strength_items(
        value_fix_candidates,
        "value_fix_priority",
        top_n=3,
        mode="value_fix"
    )

    if len(value_fix_items) == 0:
        render_info_card(
            "價值修復建議",
            "目前沒有明顯價值弱點，表示在同價格帶中，尚未出現明確的價格－價值失衡訊號。",
            border_color="#C9A227",
            bg_color="#FFFBEA"
        )
    else:
        render_highlight_block(
    "價值修復建議",
    value_fix_items,
    kind="warning"
)

st.markdown("## 核心結論")

top_indicator = comparison.sort_values("Z", ascending=False).iloc[0]["indicator"]

render_info_card(
    "市場定位結論",
    f"{own_name} 在目前價格區隔中，最具競爭力的指標為 <b>{top_indicator}</b>，整體品質定位偏向市場中上水準，建議以優勢指標作為核心行銷主軸。",
    border_color="#2F80ED",
    bg_color="#F8FAFC"
)

display_cols = [
    "indicator",
    "Z",
    "value_z",
    "W",
    "quality_strategy",
    "value_strategy",
    "indicator_reliability"
]

existing_display_cols = [c for c in display_cols if c in comparison.columns]
st.dataframe(
    comparison[existing_display_cols],
    width="stretch"
)

st.dataframe(comparison, use_container_width=True)

if dev_mode:
    st.subheader("優勢放大建議")
    st.subheader("品質放大建議")

    top_quality_amplify = comparison.sort_values(
        "quality_amplify_priority",
        ascending=False
    ).head(3)

    st.dataframe(top_quality_amplify[[
        "indicator",
        "Z",
        "W",
        "quality_amplify_priority"
    ]], use_container_width=True)

    st.subheader("弱點修復建議")
    st.subheader("價值修復建議")

top_value_fix = comparison.sort_values(
    "value_z"
).head(3)
top_value_fix = top_value_fix.sort_values(
    "value_fix_priority",
    ascending=False
).head(3)

if len(top_value_fix) == 0:
    st.info("目前沒有明顯價值弱點")
else:
    st.dataframe(top_value_fix[[
        "indicator",
        "value_z",
        "W",
        "value_fix_priority"
    ]], use_container_width=True)

dev_mode = st.sidebar.toggle("開發模式", value=False)
if dev_mode:
    st.subheader("DEBUG: products")
    st.write(products.shape)
    st.dataframe(products, width="stretch")

    st.subheader("DEBUG: reviews")
    st.write(reviews.shape)
    st.dataframe(reviews.head(10), width="stretch")

    st.subheader("DEBUG: all_scores")
    st.write(all_scores.shape)
    st.dataframe(all_scores.head(30), width="stretch")

    st.subheader("DEBUG: adjusted_score counts by product and indicator")
    st.dataframe(
        all_scores.groupby(["product_id", "indicator"])["adjusted_score"].count().reset_index(),
        width="stretch"
    )

    st.subheader("DEBUG: market_stats")
st.write(market_stats.shape)
st.dataframe(market_stats, width="stretch")

display_cols = [
    "indicator",
    "mention_count",
    "market_avg_mention",
    "adjusted_score",
    "mu",
    "sigma",
    "sigma_raw",
    "sigma_used",
    "competitor_n",
    "Z",
    "price_z",
    "value_z",
    "W",
    "low_signal_indicator",
    "indicator_reliability",
    "quality_strategy",
    "value_strategy",
    "quality_amplify_priority",
    "value_amplify_priority",
    "quality_fix_priority",
    "value_fix_priority"
]

existing_display_cols = [c for c in display_cols if c in comparison.columns]
st.dataframe(comparison[existing_display_cols], width="stretch")