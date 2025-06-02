# -*- coding: utf-8 -*-
"""
Created on Mon May 12 07:53:38 2025
@author: user
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
# 4. 三種圖表：箱型圖、散佈圖、雷達圖
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
        print(bar_df[["ratio", "price_unit", "price_total"]].dtypes)
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
# 5. 探索性資料分析：變數關聯與趨勢
# ===============================
st.header("🔍 探索性資料分析（EDA）")

eda_df = df[["age", "area", "room", "ratio", "price_unit", "price_total"]].dropna()

st.subheader("數值欄位熱力相關係數矩陣")
corr = eda_df.corr(numeric_only=True)
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="數值欄位相關係數")
st.plotly_chart(fig_corr)

st.subheader("變數與單價的關聯")

# 中文欄位對應原始欄位
eda_options = {
    "屋齡": "age",
    "面積": "area",
    "房間數": "room",
    "占比率": "ratio"
}

# 選單顯示中文，實際用欄位名稱
eda_option_zh = st.selectbox("請選擇要分析的變數", list(eda_options.keys()))
eda_option = eda_options[eda_option_zh]

# 畫圖
fig_scatter = px.scatter(
    eda_df,
    x=eda_option,
    y="price_unit",
    trendline="ols",
    title=f"{eda_option_zh} 與單價的關聯",
    labels={eda_option: eda_option_zh, "price_unit": "單價"}
)
st.plotly_chart(fig_scatter)

# 取得回歸結果並顯示R平方值
results = px.get_trendline_results(fig_scatter)
model = results.iloc[0]["px_fit_results"]
st.write(f"回歸模型 R平方值 = {model.rsquared:.4f}")

# ===============================
# 6. 使用者輸入進行預測
# ===============================
st.subheader("🔍 輸入資料 → 預測單價")

input_age = st.number_input("屋齡", min_value=10, max_value=100, value=25)
input_area = st.number_input("總面積", min_value=10, max_value=260, value=45)
input_room = st.number_input("房間數", min_value=2, max_value=10, value=3)

# 預測模型預設（需事先訓練）
# 這裡給一個預設model示意，你要自己加載或訓練模型
try:
    model
except NameError:
    # 假設有訓練好的model，可以直接載入
    # 這裡先用簡單訓練示例代替，避免程式錯誤
    from sklearn.linear_model import LinearRegression
    X = df[["age", "area", "room"]].fillna(0)
    y = df["price_unit"].fillna(0)
    model = LinearRegression().fit(X, y)

if st.button("預測"):
    input_data = pd.DataFrame([[input_age, input_area, input_room]],
                              columns=["age", "area", "room"])
    pred = model.predict(input_data)[0]
    st.success(f"🌟 預測單價為：{pred:.2f} 萬元")
