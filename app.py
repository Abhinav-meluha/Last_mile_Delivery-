# ======================================================
# 📦 Last-Mile Delivery Dashboard – FA2 (Abhinav Chanakya)
# ======================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Last-Mile Delivery Analytics", layout="wide")
st.title("🚚 Last-Mile Delivery Analytics Dashboard")
st.caption("IBCP – AI Career-Related Study | FA2 Project by Abhinav Chanakya")

# ---------- DATA LOADING ----------
DATA_PATH = "Copy of Last mile Delivery Data.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found. Please upload 'Copy of Last mile Delivery Data.csv' to the project folder.")
    st.stop()

df_raw = pd.read_csv(DATA_PATH, dtype=str)

# ---------- BASIC CLEANING ----------
df_raw.columns = df_raw.columns.str.strip()
df = df_raw.copy()

# Normalize missing values
for col in df.columns:
    df[col] = df[col].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan})

# ---------- FLEXIBLE COLUMN DETECTION ----------
variants = {
    "delivery_time": ["Delivery_Time", "Delivery Time", "Time Taken", "delivery_time"],
    "weather": ["Weather", "weather"],
    "traffic": ["Traffic", "traffic"],
    "vehicle": ["Vehicle", "vehicle", "Vehicle Type"],
    "agent_age": ["Agent_Age", "Agent Age", "Age"],
    "agent_rating": ["Agent_Rating", "Agent Rating", "Rating"],
    "area": ["Area", "area", "Region"],
    "category": ["Category", "category", "Product Category"]
}

def find_col(variants, df_cols):
    mapping = {}
    for key, options in variants.items():
        found = None
        for o in options:
            for c in df_cols:
                if c.lower().strip() == o.lower().strip():
                    found = c
                    break
            if found: break
        mapping[key] = found
    return mapping

col_map = find_col(variants, df.columns)
st.sidebar.header("🧩 Column Mapping")
for k, v in col_map.items():
    st.sidebar.write(f"• {k} → {v}")

if not col_map["delivery_time"]:
    st.error("Could not find 'Delivery_Time' column. Please check your CSV header.")
    st.stop()

# ---------- STANDARDIZE COLUMN NAMES ----------
work = pd.DataFrame()
for key in col_map:
    src = col_map[key]
    if src:
        work[key] = df[src]
    else:
        work[key] = np.nan

# Convert numeric columns
for c in ["delivery_time", "agent_age", "agent_rating"]:
    work[c] = pd.to_numeric(work[c], errors="coerce")

# Drop invalid rows
rows_before = len(work)
work.dropna(subset=["delivery_time"], inplace=True)
rows_after = len(work)

st.sidebar.markdown("### 📊 Data Summary")
st.sidebar.write(f"Rows before cleaning: {rows_before}")
st.sidebar.write(f"Rows after cleaning: {rows_after}")
st.sidebar.dataframe(work.head())

# ---------- FEATURE ENGINEERING ----------
mean_time = work["delivery_time"].mean()
std_time = work["delivery_time"].std()
work["late"] = np.where(work["delivery_time"] > (mean_time + std_time), 1, 0)
work["age_group"] = pd.cut(work["agent_age"], bins=[0,24,40,100], labels=["<25", "25–40", "40+"], include_lowest=True)

# ---------- FILTERS ----------
st.sidebar.header("🔍 Filters")
def make_filter(col):
    vals = sorted(work[col].dropna().unique().tolist())
    if vals:
        return st.sidebar.multiselect(col.capitalize(), vals, default=vals)
    else:
        return []

weather_sel = make_filter("weather")
traffic_sel = make_filter("traffic")
vehicle_sel = make_filter("vehicle")
area_sel = make_filter("area")
category_sel = make_filter("category")

filtered = work[
    work["weather"].isin(weather_sel)
    & work["traffic"].isin(traffic_sel)
    & work["vehicle"].isin(vehicle_sel)
    & work["area"].isin(area_sel)
    & work["category"].isin(category_sel)
]

st.write(f"Filtered records: {len(filtered)} / {len(work)}")

# ---------- METRICS ----------
col1, col2, col3 = st.columns(3)
col1.metric("Avg Delivery Time (min)", f"{filtered['delivery_time'].mean():.2f}")
col2.metric("Avg Agent Rating", f"{filtered['agent_rating'].mean():.2f}")
col3.metric("Late Deliveries (%)", f"{filtered['late'].mean() * 100:.1f}%")

st.markdown("---")

# ---------- VISUALS ----------
def show_bar(df, x, y, title):
    if df.empty:
        st.info(f"No data available for {title}.")
    else:
        fig = px.bar(df, x=x, y=y, title=title, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

def show_scatter(df, x, y, color, title):
    if df.empty:
        st.info(f"No data for {title}.")
    else:
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True)

# Weather vs Delivery Time
st.subheader("🌦️ Average Delivery Time by Weather")
show_bar(filtered.groupby("weather", as_index=False)["delivery_time"].mean(), "weather", "delivery_time", "Avg Delivery Time by Weather")

# Traffic vs Delivery Time
st.subheader("🚦 Average Delivery Time by Traffic")
show_bar(filtered.groupby("traffic", as_index=False)["delivery_time"].mean(), "traffic", "delivery_time", "Avg Delivery Time by Traffic")

# Vehicle Analysis
st.subheader("🚗 Vehicle Performance")
show_bar(filtered.groupby("vehicle", as_index=False)["delivery_time"].mean(), "vehicle", "delivery_time", "Avg Delivery Time by Vehicle")

# Agent Performance
st.subheader("👤 Agent Performance – Rating vs Delivery Time")
show_scatter(filtered, "agent_rating", "delivery_time", "age_group", "Agent Rating vs Delivery Time")

# Area Analysis
st.subheader("📍 Area-Wise Average Delivery Time")
show_bar(filtered.groupby("area", as_index=False)["delivery_time"].mean(), "area", "delivery_time", "Avg Delivery Time by Area")

# Category Analysis
st.subheader("📦 Category Distribution – Delivery Time")
if filtered.empty:
    st.info("No data for selected filters.")
else:
    fig = px.box(filtered, x="category", y="delivery_time", title="Delivery Time Distribution by Category")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("💡 Tip: Use the sidebar filters to explore how traffic, weather, or vehicle type affect delivery time.")
st.caption("Developed by Abhinav Chanakya | IBCP AI CRS FA2 | © 2025")
