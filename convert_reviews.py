import pandas as pd

# 讀取原始資料
df = pd.read_csv(r"民宿數據_客戶評價.csv")

# 合併優點與缺點
df["review_text"] = df["民宿優點"].fillna("") + " " + df["民宿缺點"].fillna("")

# 建立新資料表（保留客群分析需要的欄位）
reviews = pd.DataFrame({
    "product_id": "OWN_001",
    "user_id": df["user_id"],
    "companion_type": df["同行類型"],
    "nationality": df["國籍"],
    "stay_days": df["入住天數"],
    "pros": df["民宿優點"],
    "cons": df["民宿缺點"],
    "review_text": df["review_text"]
})

# 輸出
reviews.to_csv("data/reviews.csv", index=False, encoding="utf-8-sig")

print("reviews.csv 已生成（含客群欄位）")
