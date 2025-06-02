# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:53:38 2025
@author: user
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ===============================
# 1. 資料載入與基本顯示
# ===============================
st.title("房價資料分析")

# 狀態訊息
st.success('分析環境載入成功 ✅')
st.info("請使用側邊欄進行篩選與互動分析", icon='ℹ️')

# 載入資料
df = pd.read_csv("Kaohsiung.csv")

# 顯示部分資料
st.header("原始資料預覽")
st.dataframe(df.head(50))

# ===============================
# 2. 側欄條件篩選
# ===============================
st.sidebar.header("🔎 資料篩選器")
age_range = st.sidebar.slider("屋齡範圍", 1, 40, (10, 20))
room = st.sidebar.selectbox("房間數", ["All", "2", "3"])
ratio_range = st.sidebar.slider("主建物佔比範圍", 35, 100, (50, 70))

# 篩選資料
filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) & 
                 (df["ratio"] >= ratio_range[0]) & (df["ratio"] <= ratio_range[1])]
if room != "All":
    filtered_df = filtered_df[filtered_df["room"] == room]

st.subheader("篩選後的資料")
st.dataframe(filtered_df)

# ===============================
# 3. 統計摘要與欄位最大/最小值
# ===============================
st.header("統計摘要")
st.write(filtered_df.describe())

st.subheader("欄位最大/最小值")
for label, col in {
    "屋齡 (age)": "age",
    "主建物佔比 (ratio)": "ratio",
    "單價 (price_unit)": "price_unit",
    "總價 (price_total)": "price_total"
}.items():
    if col in filtered_df.columns:
        st.write(f"{label} ➤ 最小：{filtered_df[col].min():.2f}，最大：{filtered_df[col].max():.2f}")

# ===============================
# 4. 三種圖表：箱型圖、散佈圖、直方圖
# ===============================
st.header("互動式圖表分析")
tab1, tab2, tab3 = st.tabs(["📦 箱型圖", "⚫ 散佈圖", "📊 直方圖"])

with tab1:
    fig1 = px.box(filtered_df, x="room", y="price_unit", title="房間數與單價")
    st.plotly_chart(fig1)

with tab2:
    fig2 = px.scatter(filtered_df, x="age", y="price_total", color="room", title="屋齡與總價")
    st.plotly_chart(fig2)

with tab3:
    if filtered_df.empty:
        st.warning("⚠️ 篩選後無資料可供圖表分析，請調整側欄條件")
    else:
        bar_df = filtered_df.dropna(subset=["ratio", "price_unit", "price_total"])
        bar_df[["ratio", "price_unit", "price_total"]] = bar_df[["ratio", "price_unit", "price_total"]].apply(pd.to_numeric, errors='coerce')
        bar_df = bar_df.dropna(subset=["ratio", "price_unit", "price_total"])
        if bar_df.empty:
            st.warning("⚠️ 欄位含缺值，請放寬篩選條件或填補缺失資料")
        else:
            avg_df = bar_df.groupby("room")[["ratio", "price_unit", "price_total"]].mean().reset_index()
            avg_df_melted = avg_df.melt(id_vars="room", var_name="指標", value_name="平均值")
            fig_bar = px.bar(avg_df_melted, x="指標", y="平均值", color="room", barmode="group",
                             title="房間數平均表現直方圖")
            st.plotly_chart(fig_bar)

# ===============================
# 5. 探索性資料分析：變數關聯與趨勢（單變數+R平方）
# ===============================
st.header("🔍 探索性資料分析（EDA）")

eda_df = df[["age", "area", "room", "ratio", "price_unit"]].dropna()

# 中文選項對應欄位
col_name_map = {
    "屋齡": "age",
    "面積": "area",
    "房間數": "room",
    "占比率": "ratio"
}

eda_option_cn = st.selectbox("請選擇要分析的變數", list(col_name_map.keys()))
eda_option = col_name_map[eda_option_cn]

# 計算單變數線性迴歸的 R^2
X = eda_df[[eda_option]]
X = sm.add_constant(X)  # 加入截距項
y = eda_df["price_unit"]
model_ols = sm.OLS(y, X).fit()
r_squared = model_ols.rsquared

# 顯示R平方於標題上方
st.write(f"**{eda_option_cn} 與單價的線性回歸 R² = {r_squared:.4f}**")

fig_scatter = px.scatter(eda_df, x=eda_option, y="price_unit", trendline="ols",
                         title=f"{eda_option_cn} 與單價的關聯", labels={eda_option: eda_option_cn, "price_unit": "單價"})
st.plotly_chart(fig_scatter)

# ===============================
# 6. 多變數回歸模型 + R平方值顯示
# ===============================
st.subheader("多變數線性回歸模型整體R平方值")

# 預處理：只用數值欄位，去除缺值
multi_df = df[["age", "area", "room", "ratio", "price_unit"]].dropna()
X_multi = multi_df[["age", "area", "room", "ratio"]]
X_multi = sm.add_constant(X_multi)
y_multi = multi_df["price_unit"]

multi_model = sm.OLS(y_multi, X_multi).fit()
r2_multi = multi_model.rsquared

st.write(f"多變數回歸模型整體 R² = {r2_multi:.4f}")

# ===============================
# 7. 使用者輸入進行預測（用多變數線性回歸模型）
# ===============================
st.subheader("🔍 使用者輸入 → 預測單價 (多變數線性迴歸)")

input_age = st.number_input("屋齡", min_value=1, max_value=100, value=25)
input_area = st.number_input("面積", min_value=1, max_value=260, value=45)
input_room = st.number_input("房間數", min_value=1, max_value=10, value=3)
input_ratio = st.number_input("占比率", min_value=1, max_value=100, value=60)

if st.button("預測"):
    input_data = pd.DataFrame([[1, input_age, input_area, input_room, input_ratio]],
                              columns=["const", "age", "area", "room", "ratio"])
    pred = multi_model.predict(input_data)[0]
    st.success(f"🌟 預測單價為：{pred:.2f} 萬元")

