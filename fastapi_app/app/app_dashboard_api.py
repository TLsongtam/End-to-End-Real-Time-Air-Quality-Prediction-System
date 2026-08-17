import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# Cấu hình trang Streamlit
st.set_page_config(page_title="Air Quality Real-time Dashboard", layout="wide")

# 🔄 BẬT AUTO-UPDATE: Tự động chạy lại trang mỗi 10 giây (10000ms)
count = st_autorefresh(interval=10000, limit=100, key="frequent_rerun")

FASTAPI_URL = "http://127.0.0.1:8000/api/v1"


st.title("🌬️ Hệ Thống Giám Sát & Dự Báo Chất Lượng Air Quality (PM2.5)")

# Sidebar chọn trạm
#station_id = st.sidebar.radio("Chọn trạm quan đo:", [1, 2, 3, 4, 5, 6])

station_id = st.segmented_control(
    "Chọn trạm quan đo:", [1, 2, 3, 4, 5, 6],
    default=1
)


# 1. Gọi API Lịch sử từ FastAPI
try:
    res_hist = requests.get(f"{FASTAPI_URL}/stations/{station_id}/history", timeout=5)
    if res_hist.status_code == 200:
        res_json = res_hist.json()
        # Trích xuất danh sách record từ key "data" nếu response là dict
        hist_records = res_json.get("data", []) if isinstance(res_json, dict) else res_json
    else:
        hist_records = []
except Exception as e:
    hist_records = []
    st.error(f"Lỗi kết nối FastAPI Backend: {e}")

# 2. Gọi API Dự báo 24h từ FastAPI
try:
    res_fore = requests.get(f"{FASTAPI_URL}/stations/{station_id}/forecast", timeout=5)
    fore_data = res_fore.json() if res_fore.status_code == 200 else {}
except Exception as e:
    fore_data = {}

# HIỂN THỊ METRICS THỜI GIAN THỰC
if hist_records:
    df_hist = pd.DataFrame(hist_records)
    latest_rec = df_hist.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PM2.5 Hiện tại", f"{latest_rec.get('PM2_5', 'N/A')} µg/m³")
    col2.metric("Nhiệt độ", f"{latest_rec.get('Temperature', 'N/A')} °C")
    col3.metric("Độ ẩm", f"{latest_rec.get('Humidity', 'N/A')} %")
    col4.metric("Trạng thái Server", "🟢 Realtime" if count else "Offline")

    # BIỂU ĐỒ LỊCH SỬ PM2.5
    st.subheader("📊 Lịch sử PM2.5 (48 giờ qua)")
    fig_hist = px.line(df_hist, y="PM2_5", title=f"Trạm {station_id} - Dữ liệu thực tế")
    st.plotly_chart(fig_hist, use_container_width=True)


# BIỂU ĐỒ DỰ BÁO 24H
if fore_data and "forecast_24h" in fore_data:
    st.subheader("🔮 Dự báo PM2.5 (24 giờ tiếp theo)")
    preds = fore_data["forecast_24h"]
    df_fore = pd.DataFrame({"Giờ thứ": [f"+{i+1}h" for i in range(len(preds))], "PM2.5 Dự báo": preds})
    fig_fore = px.line(df_fore, x="Giờ thứ", y="PM2.5 Dự báo", markers=True)
    st.plotly_chart(fig_fore, use_container_width=True)