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

# Khởi tạo CSS customized để làm mượt giao diện và sửa lỗi chìm chữ trên Cloud
st.markdown("""
    <style>
    .main-title { font-size:32px !important; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size:18px !important; text-align: center; color: #4B5563; margin-bottom: 20px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 6px solid #1E3A8A; color: #0F172A; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); }
    /* Đảm bảo chữ trên các Tab luôn hiển thị rõ ràng */
    .stTabs button { color: inherit !important; font-weight: 600 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 ỨNG DỤNG PHÂN TÍCH CẤU TRÚC THỊ TRƯỜNG CHỨNG KHOÁN VIỆT NAM BẰNG PCA FROM SCRATCH</div>', unsafe_allow_html=True)

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
        
        # ĐỌC FILE BẰNG UTF-8 VÀ LẤY THEO VỊ TRÍ CỘT (Khắc phục hoàn toàn lỗi font chữ hệ thống)
        try:
            df_raw = pd.read_csv(file_path, encoding='utf-8')
        except:
            df_raw = pd.read_csv(file_path, encoding='utf-8-sig') # Phòng hờ file có BOM
            
        # Lấy cột 0 (Ngày) và cột 1 (Giá phối/Lần cuối) dựa theo chỉ số vị trí, không quan tâm tên chữ
        df_temp = df_raw.iloc[:, [0, 1]].copy()
        df_temp.columns = ['Date', ticker]
        
        # Ép định dạng Ngày/Tháng/Năm chuẩn xác
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Làm sạch giá tiền
        if df_temp[ticker].dtype == 'object':
            df_temp[ticker] = df_temp[ticker].astype(str).str.replace(',', '')
            df_temp[ticker] = df_temp[ticker].str.replace('%', '') 
            df_temp[ticker] = df_temp[ticker].str.strip() 
            
        df_temp[ticker] = pd.to_numeric(df_temp[ticker], errors='coerce')
            
        if df_merged is None:
            df_merged = df_temp
        else:
            df_merged = pd.merge(df_merged, df_temp, on='Date', how='outer')
            
    if df_merged is not None:
        # Loại bỏ dòng lỗi nếu có
        df_merged = df_merged.dropna(subset=['Date'])
        df_merged = df_merged.sort_values('Date').set_index('Date')
        
        # Đảm bảo index ngày tháng không bị dính múi giờ tự động
        df_merged.index = pd.to_datetime(df_merged.index).tz_localize(None)
        
        df_merged = df_merged.astype(float)
        df_merged = df_merged.ffill().bfill()
        
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

    # Lọc dữ liệu theo ngày đã chọn
    df_filtered_prices = df_prices.loc[str(start_date):str(end_date)]

    # Chỉ giữ lại các cột thực sự là kiểu số (float/int) để tính toán
    df_filtered_prices = df_filtered_prices.select_dtypes(include=[np.number])

    # Tính toán tỷ suất sinh lợi Logarit hằng ngày từ dữ liệu đã lọc
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
        
        fig_heat = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale=DIVERGING_HEX_SCALE, 
            labels=dict(color="Hệ số tương quan")
        )
        # CHỖ NÀY: Đảm bảo thụt lề bằng đúng với dòng fig_heat phía trên
        fig_heat.update_layout(
            height=700, 
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white" 
        )
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
                margin=dict(l=20, r=20, t=20, b=20), height=400,
                template="plotly_white"
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
            xaxis_title="Thời gian", 
            yaxis_title="Tỷ suất sinh lợi tích lũy",
            hovermode="x unified", 
            template="plotly_white" 
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
        
        fig_bar_pc2 = px.bar(
            df_loadings_pc2,
            x="Mã Cổ Phiếu", y="Trọng số Loading",
            color="Trọng số Loading",
            color_continuous_scale=DIVERGING_HEX_SCALE, 
            title="Sức ảnh hưởng phân hóa dòng tiền của các cổ phiếu cấu thành PC2"
        )
        fig_bar_pc2.update_layout(
            template="plotly_white", 
            xaxis={'categoryorder':'total descending'}
        )
        st.plotly_chart(fig_bar_pc2, use_container_width=True)
        
        # Khối kết luận biện luận mở rộng tích hợp PC1 & PC2 (Bản nâng cấp thực tế cho HUB)
        st.markdown("""
        <div class="card" style="line-height: 1.8; text-align: justify; padding: 20px; border-radius: 12px; background-color: #F8FAFC; border-left: 6px solid #1E3A8A;">
        <h4 style="color: #1E3A8A; margin-top: 0; font-size: 20px; font-weight: 700; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px;">💡 ĐÁNH GIÁ VÀ BÀI HỌC KINH TẾ TỪ MÔ HÌNH PCA (GÓC NHÌN ĐẦU TƯ THỰC TẾ)</h4>
        
        <p><strong>1. Nhìn từ PC1: Đầu tư ở Việt Nam, xu hướng chung quyết định tất cả</strong><br>
        Nhìn vào con số tương quan lên tới <strong>91.92%</strong> giữa đường PC1 và chỉ số VN30 thực tế, ta thấy một sự thật là: Ở thị trường chứng khoán Việt Nam, xu hướng chung của thị trường (bản chất là dòng tiền lớn, lãi suất, vĩ mô) quyết định đến hơn 90% sự tăng giảm của cổ phiếu. Khi thị trường vào sóng tăng hoặc sụt giảm, tâm lý bầy đàn xuất hiện và kéo gần như tất cả các mã đi chung một hướng. Đối với người quản trị danh mục, con số này là một lời cảnh báo: Khi thị trường chung sập, việc bạn đa dạng hóa danh mục bằng cách mua nhiều mã khác nhau trong rổ VN30 gần như vô tác dụng, vì lúc đó rủi ro hệ thống đã bao trùm toàn bộ.</p>
        
        <p><strong>2. Nhìn từ PC2: Câu chuyện luân chuyển dòng tiền và "Cán cân ngành"</strong><br>
        Nếu như PC1 nói về xu hướng chung, thì PC2 lại vạch trần câu chuyện phân hóa ngành và cuộc chơi luân chuyển dòng tiền của các mập. Biểu đồ trọng số (Loadings) của PC2 đã chia rổ VN30 thành hai chiến tuyến đối lập rất rõ ràng:
        <ul>
            <li><strong>Phía Dương (Nhóm phòng thủ, sản xuất, năng lượng):</strong> Gồm các ông lớn có dòng tiền mặt cực khỏe như <code>GAS</code>, <code>PLX</code>, <code>GVR</code>, <code>VNM</code> và hai bank quốc doanh trụ cột là <code>VCB</code>, <code>BID</code>.</li>
            <li><strong>Phía Âm (Nhóm chu kỳ, bất động sản và bank tư nhân):</strong> Bị thống trị bởi họ nhà Vin (<code>VHM</code>, <code>VIC</code>, <code>VRE</code>) và các ngân hàng thương mại cổ phần tư nhân nhạy cảm với tín dụng như <code>TCB</code>, <code>HDB</code>.</li>
        </ul>
        Nhìn vào đường đồ thị PC2, giai đoạn giữa năm 2025 nó cắm đầu đi xuống vì dòng tiền lúc đó chê nhóm bất động sản và bank tư nhân để chạy sang trú ẩn ở nhóm năng lượng, sản xuất (Phía Dương thắng thế). Ngược lại, từ cuối năm 2025 đến đầu năm 2026, đường PC2 lại dựng đứng lên. Điều này chứng minh dòng tiền đầu cơ đã quay xe, rút mạnh khỏi nhóm phòng thủ để lao vào đánh sóng hồi của bất động sản và tài chính tư nhân (Phía Âm bùng nổ). Bản chất của PC2 chính là thước đo xem dòng tiền thông minh đang chảy vào túi ngành nào.</p>
        
        <p><strong>3. Ứng dụng thực tế để thiết kế danh mục đầu tư</strong><br>
        Từ thuật toán PCA tự dựng này, chúng ta rút ra một mẹo xương máu khi làm danh mục đầu tư: Đừng bao giờ mua hai cổ phiếu nằm cùng một phía của PC2 (ví dụ đã mua <code>VHM</code> lại còn mua thêm <code>TCB</code>, hoặc đã ôm <code>GAS</code> lại mua thêm <code>PLX</code>). Vì khi dòng tiền rút khỏi nhóm đó, danh mục của bạn sẽ bị vạ lây cả đôi. Cách đi tiền khôn ngoan là bắt cặp chéo giữa một mã phía Dương (như <code>VCB</code> hoặc <code>GAS</code>) with một mã phía Âm (như <code>TCB</code> hoặc <code>VHM</code>). Sự bù trừ này giúp danh mục luôn có chỗ dựa vững chắc bất kể dòng tiền thị trường có xoay vòng thế nào đi chăng nữa.</p>
        </div>
        """, unsafe_allow_html=True)
    