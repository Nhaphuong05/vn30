import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import unicodedata

# 1. CẤU HÌNH GIAO DIỆN HIỆN ĐẠI
st.set_page_config(
    page_title="VN30 Market Structure PCA Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo CSS customized để làm mượt giao diện
st.markdown("""
    <style>
    .main-title { font-size:32px !important; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size:18px !important; text-align: center; color: #4B5563; margin-bottom: 20px; }
    .card { background-color: #F3F4F6; padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #1E3A8A; color: #1F2937; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Ứng dụng Phân tích cấu trúc thị trường chứng khoán Việt Nam bằng thuật toán PCA from Scratch</div>', unsafe_allow_html=True)

# Khối thông tin nhóm ở Sidebar
st.sidebar.markdown("### 👥 NGƯỜI THỰC HIỆN")
st.sidebar.markdown("- **Nguyễn Thị Nhã Phương**")

DATA_FOLDER = "VN30"

def clean_ticker_name(filename):
    filename_normalized = unicodedata.normalize('NFC', filename)
    ticker = filename_normalized.replace("Dữ liệu Lịch sử ", "").replace(".csv", "").replace(" ", "")
    return ticker

@st.cache_data
def load_raw_data(folder):
    file_list = glob.glob(os.path.join(folder, "*.csv"))
    if not file_list:
        return None
    
    df_merged = None
    for file_path in file_list:
        filename = os.path.basename(file_path)
        ticker = clean_ticker_name(filename)
        
        df_temp = pd.read_csv(file_path)[['Ngày', 'Lần cuối']].copy()
        df_temp.columns = ['Date', ticker]
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%d/%m/%Y')
        if df_temp[ticker].dtype == 'object':
            df_temp[ticker] = df_temp[ticker].str.replace(',', '').astype(float)
            
        if df_merged is None:
            df_merged = df_temp
        else:
            df_merged = pd.merge(df_merged, df_temp, on='Date', how='outer')
            
    df_merged = df_merged.sort_values('Date').set_index('Date').ffill().bfill()
    return df_merged

# Tải dữ liệu thô
df_prices = load_raw_data(DATA_FOLDER)

if df_prices is None:
    st.error(f"⚠️ Không tìm thấy dữ liệu CSV tại thư mục: '{DATA_FOLDER}'. Hãy đảm bảo thư mục này nằm ngang hàng với file app.py.")
else:
    # 2. SIDEBAR - ĐIỀU KHIỂN ĐỘNG
    st.sidebar.header("⚙️ BỘ LỌC VÀ CẤU HÌNH MÔ HÌNH")
    min_date = df_prices.index.min().to_pydatetime()
    max_date = df_prices.index.max().to_pydatetime()
    
    start_date, end_date = st.sidebar.date_input(
        "Chọn khoảng thời gian phân tích:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    steps_qr = st.sidebar.slider("Số vòng lặp thuật toán QR", 100, 1500, 500, step=100)
    target_var = st.sidebar.slider("Ngưỡng phương sai giải thích tích lũy (%)", 50, 95, 90, step=5) / 100.0

    # Lọc dữ liệu theo ngày
    df_filtered_prices = df_prices.loc[str(start_date):str(end_date)]
    df_returns = np.log(df_filtered_prices / df_filtered_prices.shift(1)).dropna()

    # 3. TOÁN PCA FROM SCRATCH 
    stocks_returns = df_returns.drop(columns=['VN30'], errors='ignore')
    X = stocks_returns.values
    features = stocks_returns.columns

    # Chuẩn hóa Z-score
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    cov_matrix = (X_scaled.T @ X_scaled) / (X_scaled.shape[0] - 1)

    # Thuật toán QR Phân rã trị riêng tự dựng
    A_k = cov_matrix.copy()
    V = np.eye(cov_matrix.shape[0])
    for _ in range(steps_qr):
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        V = V @ Q

    eigenvalues, eigenvectors = np.diag(A_k), V
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # MẢNG MÀU HEX ĐỐI LẬP CHUẨN (Xanh dương - Trắng - Đỏ) KHÔNG DÙNG TÊN CHUỖI CỦA PLOTLY
    DIVERGING_HEX_SCALE = [[0.0, '#005f73'], [0.3, '#94d2bd'], [0.5, '#e9d8a6'], [0.7, '#ee9b00'], [1.0, '#ae2012']]

    # 4. THIẾT KẾ CÁC TABS DASHBOARD
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tổng Quan Thị Trường", 
        "🧮 Tương Quan Hệ Số", 
        "🧬 Đặc Trưng Nhân Tố PCA", 
        "🎯 Dự Phóng Nhân Tố PC1 & PC2"
    ])

    with tab1:
        st.markdown("### 💸 Chỉ số Giá & Biến động rổ VN30")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="Tổng số ngày giao dịch lọc", value=f"{df_returns.shape[0]} ngày")
        with kpi2:
            st.metric(label="Số lượng tài sản cấu thành", value=f"{stocks_returns.shape[1]} Cổ phiếu")
        with kpi3:
            pc1_ratio = explained_variance_ratio[0] * 100
            st.metric(label="Sức mạnh giải thích của PC1", value=f"{pc1_ratio:.2f}%", delta="Nhân tố hệ thống")
            
        st.markdown("#### Bảng tra cứu Tỷ suất sinh lợi Logarit đầu vào")
        st.dataframe(df_returns, use_container_width=True)

    with tab2:
        st.markdown("### 🧮 Ma trận Tương quan Tỷ suất sinh lợi (Interactive Heatmap)")
        corr_matrix = stocks_returns.corr()
        
        # FIX: Dùng trực tiếp mảng Hex màu cố định để triệt tiêu lỗi hệ thống chuỗi tên màu
        fig_heat = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=DIVERGING_HEX_SCALE, 
            labels=dict(color="Hệ số tương quan")
        )
        fig_heat.update_layout(height=700, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("### 🧬 Kết quả trích xuất Trị riêng & Phương sai tích lũy")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Bảng xếp hạng 10 Thành phần chính đầu tiên")
            df_eigen = pd.DataFrame({
                "Thành phần chính": [f"PC{i+1}" for i in range(len(eigenvalues))],
                "Trị riêng (Eigenvalue)": eigenvalues,
                "Phương sai đơn lẻ giải thích (%)": explained_variance_ratio * 100,
                "Phương sai tích lũy giải thích (%)": cumulative_variance * 100
            })
            st.dataframe(df_eigen.head(10), use_container_width=True)
            
            k_needed = np.argmax(cumulative_variance >= target_var) + 1
            st.warning(f"💡 Nhận xét: Hệ thống cần giữ lại ít nhất **{k_needed} Thành phần chính** để đại diện cho {target_var*100}% thông tin của rổ VN30.")

        with col_right:
            st.markdown("#### Biểu đồ Phương sai tích lũy tương tác (Scree Plot)")
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Scatter(
                x=[i+1 for i in range(len(cumulative_variance))],
                y=cumulative_variance * 100,
                mode='lines+markers',
                name='Phương sai tích lũy',
                line=dict(color='#10B981', width=3)
            ))
            fig_scree.add_trace(go.Scatter(
                x=[1, len(cumulative_variance)],
                y=[target_var * 100, target_var * 100],
                mode='lines',
                name=f'Ngưỡng {target_var*100}% đặt ra',
                line=dict(color='red', dash='dash')
            ))
            fig_scree.update_layout(
                xaxis_title="Số lượng Thành phần chính (PCs)",
                yaxis_title="Tỷ lệ phương sai tích lũy (%)",
                margin=dict(l=20, r=20, t=20, b=20), height=400
            )
            st.plotly_chart(fig_scree, use_container_width=True)

    with tab4:
        st.markdown("### 🎯 Dự Phóng Nhân Tố PC1 & PC2 so với VN30 Thực tế")
        
        # Tích hợp toán cho cả PC1 và PC2
        pc1_eigenvector = eigenvectors[:, 0]
        pc2_eigenvector = eigenvectors[:, 1]
        
        pc1_returns = np.dot(X_scaled, pc1_eigenvector)
        pc2_returns = np.dot(X_scaled, pc2_eigenvector)
        
        df_analysis = pd.DataFrame(index=df_returns.index)
        df_analysis['PC1_Returns'] = pc1_returns
        df_analysis['PC2_Returns'] = pc2_returns
        
        if 'VN30' in df_returns.columns:
            df_analysis['VN30_Returns'] = df_returns['VN30'].values
        else:
            df_analysis['VN30_Returns'] = df_returns.mean(axis=1).values

        # Khắc phục Sign Ambiguity cho PC1 nhằm bám sát đồ thị gốc
        correlation_pc1 = df_analysis['PC1_Returns'].corr(df_analysis['VN30_Returns'])
        if correlation_pc1 < 0:
            df_analysis['PC1_Returns'] = -df_analysis['PC1_Returns']
            pc1_eigenvector = -pc1_eigenvector
            correlation_pc1 = -correlation_pc1

        # Điều chỉnh lại biên độ dao động (Volatility Rescaling) về chuẩn VN30
        vn30_std = df_analysis['VN30_Returns'].std()
        df_analysis['PC1_Adjusted'] = df_analysis['PC1_Returns'] * (vn30_std / df_analysis['PC1_Returns'].std())
        df_analysis['PC2_Adjusted'] = df_analysis['PC2_Returns'] * (vn30_std / df_analysis['PC2_Returns'].std())

        # Tính toán Lợi suất lũy kế tích lũy sinh lợi (Cumulative Returns)
        df_cum = pd.DataFrame(index=df_analysis.index)
        df_cum['VN30 Index (Thực tế)'] = np.exp(df_analysis['VN30_Returns'].cumsum()) - 1
        df_cum['PC1 Index (Nhân tố Thị trường)'] = np.exp(df_analysis['PC1_Adjusted'].cumsum()) - 1
        df_cum['PC2 Index (Nhân tố Phân hóa ngành)'] = np.exp(df_analysis['PC2_Adjusted'].cumsum()) - 1

        # ---- ĐỒ THỊ 1: CHUỖI ĐƯỜNG LŨY KẾ ĐA NHÂN TỐ ----
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=df_cum.index, y=df_cum['VN30 Index (Thực tế)'], mode='lines', name='VN30 Index (Thực tế)', line=dict(color='#2B2D42', width=2.5)))
        fig_line.add_trace(go.Scatter(x=df_cum.index, y=df_cum['PC1 Index (Nhân tố Thị trường)'], mode='lines', name='PC1 Index (Xu hướng)', line=dict(color='#EF233C', width=2, dash='dash')))
        fig_line.add_trace(go.Scatter(x=df_cum.index, y=df_cum['PC2 Index (Nhân tố Phân hóa ngành)'], mode='lines', name='PC2 Index (Phân hóa)', line=dict(color='#0077B6', width=2, dash='dot')))
        
        fig_line.update_layout(
            title="Đồ thị lũy kế đa nhân tố: VN30 Index vs PC1 vs PC2",
            xaxis_title="Thời gian", yaxis_title="Tỷ suất sinh lợi tích lũy",
            hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Khu vực chỉ số tương quan cặp song hành
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Hệ số tương quan Hệ thống (PC1 vs VN30 Real):", value=f"{correlation_pc1:.4f}")
        with m2:
            correlation_pc2 = df_analysis['PC2_Returns'].corr(df_analysis['VN30_Returns'])
            st.metric(label="Hệ số tương quan Trực giao (PC2 vs VN30 Real):", value=f"{correlation_pc2:.4f}", delta="Tính chất độc lập dòng tiền")

        # ---- ĐỒ THỊ 2: TRỌNG SỐ PHÂN BỔ NHÓM NGÀNH ĐỐI LẬP (PC2 LOADINGS) ----
        st.markdown("#### 📊 Phân tách cấu trúc Trọng số (Loadings) của Thành phần chính PC2")
        st.markdown("Biểu đồ thể hiện sự đối lập giữa các nhóm cổ phiếu tạo nên lực đẩy phân hóa cho nhân tố PC2.")
        
        df_loadings_pc2 = pd.DataFrame({
            "Mã Cổ Phiếu": features,
            "Trọng số Loading": pc2_eigenvector
        }).sort_values(by="Trọng số Loading", key=abs, ascending=False)
        
        # FIX: Ép bảng màu biểu đồ cột sang mảng Hex cố định để triệt tiêu lỗi
        fig_bar_pc2 = px.bar(
            df_loadings_pc2,
            x="Mã Cổ Phiếu", y="Trọng số Loading",
            color="Trọng số Loading",
            color_continuous_scale=DIVERGING_HEX_SCALE, 
            title="Sức ảnh hưởng phân hóa dòng tiền của các cổ phiếu cấu thành PC2"
        )
        fig_bar_pc2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis={'categoryorder':'total descending'}
        )
        st.plotly_chart(fig_bar_pc2, use_container_width=True)
        
        # Khối kết luận biện luận mở rộng tích hợp PC1 & PC2
        st.markdown("""
        <div class="card">
        <strong>💡 Kết luận:</strong><br>
        1. <strong>Sức mạnh của PC1 (Market Driver):</strong> Tương quan xấp xỉ 92% chứng minh cấu trúc rổ VN30 bị chi phối nặng nề bởi tính rủi ro hệ thống toàn thị trường. Khi nhân tố vĩ mô tốt, toàn bộ các cổ phiếu có trọng số PC1 đồng thuận kéo chỉ số đi lên.<br>
        2. <strong>Bản chất dịch chuyển dòng tiền ở PC2 (Sector Rotation):</strong> Biểu đồ cột Loadings của PC2 bộc lộ rõ nét trạng thái "cán cân". Một đầu cực dương là nhóm cổ phiếu Năng lượng/Tiêu dùng lớn (như GAS, PLX, GVR, VNM), đầu cực âm đối trọng lại chính là nhóm Bất động sản thương mại (họ Vin gồm VHM, VIC, VRE) kết hợp cùng một vài mã Ngân hàng tư nhân. Sự đổi màu đột ngột trên đường đồ thị PC2 giai đoạn cuối năm 2025 - đầu năm 2026 chính là minh chứng thực tế cho việc dòng tiền rút từ nhóm cổ phiếu chu kỳ này để quay vòng sang nhóm phòng thủ chu kỳ khác.
        </div>
        """, unsafe_allow_html=True)