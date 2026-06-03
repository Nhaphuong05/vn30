# 📊 VN30 Market Structure Analysis & Factor Investment Modeling via PCA From Scratch

An unsupervised machine learning and quantitative finance project designed to extract hidden market factors, analyze systemic risk, and track smart money sector rotation within the Vietnamese Stock Market (VN30 Index) using a Principal Component Analysis (PCA) model built completely from scratch.

## 🌐 Live Dashboard Application
* **Streamlit Cloud Web App:** [https://vn30project.streamlit.app/]((https://vn30-pca.streamlit.app/))
---

## 📑 1. Tổng Quan Dữ Liệu Nguồn (Data Specifications)

### 📊 Đặc điểm tập dữ liệu
Dự án sử dụng tập dữ liệu lịch sử tài chính đa biến (Multivariate Time-series Financial Data) dạng chuỗi thời gian của thị trường chứng khoán Việt Nam. 
* **Tài sản nghiên cứu:** Bao gồm **30 mã cổ phiếu thành phần** thuộc rổ chỉ số VN30 (các cổ phiếu có vốn hóa lớn nhất và tính thanh khoản cao nhất sàn HOSE) cùng với **chỉ số tham chiếu VN30-Index**.
* **Danh sách chi tiết 31 cấu phần dữ liệu:** `ACB`, `BCM`, `BID`, `CTG`, `DGC`, `FPT`, `GAS`, `GVR`, `HDB`, `HPG`, `LPB`, `MBB`, `MSN`, `MWG`, `PLX`, `SAB`, `SHB`, `SSB`, `SSI`, `STB`, `TCB`, `TPB`, `VCB`, `VHM`, `VIB`, `VIC`, `VJC`, `VNM`, `VPB`, `VRE` và chỉ số tổng `VN30`.
* **Định dạng tệp lưu trữ:** Mỗi mã tài sản được lưu trữ độc lập dưới dạng một tệp định dạng phẳng `.csv` (Comma-Separated Values).

### ⏱️ Khung thời gian & Tần suất trích xuất
* **Khoảng thời gian phân tích (Date Range):** Từ **29/04/2025 đến 29/04/2026**.
* **Tần suất quan sát (Data Frequency):** Tần suất hằng ngày (`Daily`).
* **Tổng số mẫu dữ liệu (Total Trade Days):** Khung thời gian quét trích xuất chính xác chuỗi dữ liệu gồm **248 ngày giao dịch thực tế** (loại trừ các ngày cuối tuần Thứ Bảy, Chủ Nhật và các ngày nghỉ lễ theo quy định của Ủy ban Chứng khoán Nhà nước Việt Nam).

### 🌐 Nguồn gốc & Phương pháp thu thập
* **Nền tảng cung cấp dữ liệu:** Toàn bộ tệp số liệu được trích xuất từ dữ liệu lịch sử (Historical Data) của cổng thông tin tài chính toàn cầu **Investing.com**.
* **Đặc điểm cấu trúc tệp thô từ Investing:** Mỗi tệp `.csv` tải về sở hữu cấu trúc bảng gồm 7 cột thông tin chính:
  1. `Ngày` (`Date`): Định dạng chuỗi ngày tháng dạng `DD/MM/YYYY`.
  2. `Lần cuối` (`Close`): Giá đóng cửa của phiên giao dịch (Biến số cốt lõi được dự án lựa chọn để tính tỷ suất sinh lợi toán học).
  3. `Mở` (`Open`): Giá mở cửa của phiên giao dịch.
  4. `Cao` (`High`): Mức giá cao nhất ghi nhận trong ngày.
  5. `Thấp` (`Low`): Mức giá thấp nhất ghi nhận trong ngày.
  6. `KL` (`Volume`): Khối lượng giao dịch (được ký hiệu dạng chuỗi văn bản rút gọn như `M` cho triệu cổ phiếu, `K` cho ngàn cổ phiếu).
  7. `% Thay đổi` (`% Chg`): Tỷ lệ phần trăm biến động so với phiên trước đó.

---

## ⚙️ 2. Cấu Trúc Tổ Chức Thư Mục Dự Án (Project Architecture)

Dự án được phân cấp mô-đun hóa đồng bộ theo đúng chuẩn cấu trúc triển khai mã nguồn mở Git/GitHub và Streamlit Cloud như sau:

```text
vn30/                              # Thư mục gốc dự án (Repository Root)
├── .streamlit/
│   └── config.toml                # Tệp cấu hình theme, giao diện và cổng port của Streamlit
├── .vscode/
│   └── settings.json              # Cấu hình môi trường làm việc của trình soạn thảo VS Code
├── VN30/                          # THƯ MỤC CHỨA DỮ LIỆU NGUỒN (Data Directory)
│   ├── Dữ liệu Lịch sử ACB.csv
│   ├── Dữ liệu Lịch sử FPT.csv
│   ├── Dữ liệu Lịch sử VN 30.csv  # File tổng chỉ số tham chiếu chứa khoảng trắng lệch chuẩn
│   └── ... (đủ 31 file dữ liệu tài chính gốc .csv)
├── .gitignore                     # Tệp cấu hình loại trừ các tệp rác hệ thống khi push Git
├── VN30.ipynb                     # File Notebook (Jupyter) - Nơi nháp, thử nghiệm thuật toán và đối chiếu thư viện
├── app.py                         # MÃ NGUỒN CHÍNH (Production Code) - Khởi chạy Dashboard Streamlit
└── requirements.txt               # Danh sách các thư viện Python bắt buộc cho môi trường Cloud
```
## 🛠️ 3. Quy Trình Tiền Xử Lý Dữ Liệu Bộ Lọc Vạn Năng (Data Preprocessing Pipeline)

Để đảm bảo tính dừng (stationarity) của chuỗi tài sản tài chính và loại bỏ các xung đột hệ thống giữa môi trường Local (Windows) và Server Deployment (Linux), dự án thiết lập quy trình xử lý dữ liệu thô nghiêm ngặt trong hàm `load_raw_data()`:

1. **Chuẩn hóa Encoding & Khắc phục lỗi font hệ thống:** Sử dụng thư viện `unicodedata` đưa các file dạng chuỗi ký tự tiếng Việt có dấu về chuẩn NFC (`utf-8`/`utf-8-sig`/`latin1`) giúp hệ điều hành Linux quét thư mục mượt mà không bị lỗi crash tên file.
2. **Thuật toán trích xuất Ticker thông minh:** Thiết lập bộ lọc cắt tỉa chuỗi dựa trên phương thức `.split()`. Tự động phát hiện và gộp các lỗi khoảng trắng tên file tổng (Ví dụ: Chuyển đổi tên file lỗi hệ thống từ `"VN 30"` hoặc số `"30"` về chuẩn duy nhất `"VN30"`), ngăn chặn việc lọt file chỉ số tổng vào rổ tính toán ma trận cổ phiếu thành phần.
3. **Chuẩn hóa cấu trúc dấu phân tách (Delimiter Auto-Detection):** Cấu hình lệnh `pd.read_csv(..., sep=None, engine='python')` giúp hệ thống tự động nhận diện đúng cấu trúc file CSV dù được xuất từ Excel bằng dấu phẩy `,` hay dấu chấm phẩy `;`.
4. **Xử lý số liệu giá lỗi định dạng (Currency String Cleaning):** Ép toàn bộ dữ liệu cột giá về chuỗi văn bản (`str`), dùng regex bóc tách và loại bỏ triệt để dấu phẩy `,` phân cách hàng ngàn của định dạng số quốc tế (Ví dụ: Giá VN30 `2,022.75` hay VNM `60,900.0`), trước khi chuyển đổi sang dạng số thực tinh khiết bằng `pd.to_numeric(..., errors='coerce')` nhằm bảo vệ dữ liệu khỏi hiện tượng biến thành rỗng (`NaN`).
5. **Đồng bộ hóa chuỗi thời gian tăng dần:** Áp dụng cơ chế `.ffill().bfill()` để điền dữ liệu trống cho các ngày lệch pha giao dịch (do mất thanh khoản tạm thời hoặc ngày lễ) và ép index về định dạng ngày tháng chuẩn. Thực hiện sắp xếp chỉ mục ngày tháng tăng dần `sort_index(ascending=True)` trước khi tính tỷ suất sinh lợi Logarit hằng ngày để đảm bảo quy luật toán học không bị đảo ngược.
6. **Tỷ suất sinh lợi Logarit (Daily Log-Returns):** Chuyển đổi chuỗi giá trị tuyệt đối sang tỷ suất sinh lợi liên tục hằng ngày:

$$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

---

## 🧮 4. Thuật Toán PCA Toán Học "From Scratch" (No Scikit-Learn)

Mô hình học máy không giám sát (Unsupervised Learning) được xây dựng hoàn toàn dựa trên các phép toán đại số tuyến tính cơ bản với thư viện `NumPy`:

### Bước 4.1: Chuẩn hóa dữ liệu (Z-score Standardization)

Đưa tỷ suất sinh lợi của 30 cổ phiếu về cùng một thang đo nhằm loại bỏ ảnh hưởng do biên độ biến động (Volatility) cá biệt của từng mã tài sản:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma + \epsilon}$$

*(Trong đó epsilon = 10^(-8) là hệ số an toàn bảo vệ, tránh lỗi chia cho số 0 mẫu số khi cổ phiếu đứng giá bất động).*

### Bước 4.2: Khởi tạo Ma trận hiệp phương sai

Về mặt toán học, ma trận hiệp phương sai tính trên dữ liệu đã chuẩn hóa Z-score thực chất chính là **Ma trận Tương quan (Correlation Matrix)**. Phương pháp này giúp trao cơ hội đóng góp thông tin công bằng cho mọi cổ phiếu, tránh việc các mã có thị giá rất lớn chi phối hoàn toàn mô hình:

$${\Sigma} = \frac{X_{\text{scaled}}^T \cdot X_{\text{scaled}}}{n - 1}$$

### Bước 4.3: Phân rã Trị riêng & Vectơ riêng bằng thuật toán Vòng lặp QR tự dựng

Thay vì gọi hàm thư viện đóng gói sẵn, dự án tự lập trình thuật toán lặp phân rã ma trận đối xứng để tìm cặp trị riêng (Eigenvalues) và vectơ riêng (Eigenvectors) với 500 vòng lặp nhằm đưa ma trận về dạng đường chéo hội tụ, cho độ chính xác khớp với thư viện `Numpy` đến tầng thập phân thứ 15:

$$\text{For } k = 1 \dots 500: \quad A_k = Q_k \cdot R_k \implies A_{k+1} = R_k \cdot Q_k$$

---

## 📈 5. Kết Quả Định Lượng Thực Tế & Đặc Trưng Nhân Tố

### 🧬 Sức mạnh giải thích phương sai tích lũy (Scree Plot Metrics)

* **Thành phần chính thứ nhất (PC1):** Sở hữu trị riêng (Eigenvalue) vượt trội **11.09**, một mình giải thích đến **36.83%** toàn bộ biến động vĩ mô của rổ chỉ số VN30.
* **Thành phần chính thứ hai (PC2):** Trị riêng đạt **2.65**, giải thích thêm **8.82%** biến động thị trường.
* **Ngưỡng tối ưu hóa chiều (Dimensionality Reduction):** Hệ thống tính toán chỉ ra rằng thị trường VN30 có độ phân hóa cao, cần giữ lại đúng **19 thành phần chính** đầu tiên để đạt được ngưỡng **90%** năng lực giải thích thông tin gốc (Giảm chiều từ 30 biến xuống 19 nhân tố).

### 🏛️ Cấu trúc Trọng số cấu thành Nhân tố (Factor Loadings)

* **PC1 - Nhân tố hệ thống (Thị trường chung):** Top các cổ phiếu sở hữu trọng số lớn nhất trong PC1 bao gồm: `TPB` (0.242), `TCB` (0.238), `VPB` (0.237), `CTG` (0.236), `MBB` (0.235), và `VIB` (0.230). Tất cả hệ số đều mang dấu dương và có độ phân bổ cực kỳ đồng đều. Điều này chứng minh **Ngành Ngân hàng** đóng vai trò là "Trực trưởng" (Market Driver) dẫn dắt rủi ro hệ thống của rổ chỉ số VN30.

---

## 🎯 6. Kết Luận & Khuyến Nghị Thực Chiến (Conclusions & Investment Insights)

Dự án trực quan hóa chuỗi đồ thị lũy kế đa nhân tố trên Dashboard và rút ra 3 bài học kinh tế cốt lõi cho một Nhà quản trị danh mục định lượng chuyên nghiệp (Portfolio Manager):

### 1️⃣ Khống chế Rủi ro Hệ thống thông qua Nhân tố PC1 (Market-Beta Hedging)

Tại thị trường chứng khoán Việt Nam, xu hướng chung của thị trường (chịu sự chi phối bởi dòng tiền lớn, chính sách tiền tệ, lãi suất và các biến số vĩ mô) quyết định hơn 1/3 sự tăng giảm của các cổ phiếu riêng lẻ. Khi rủi ro hệ thống xuất hiện (Thị trường bước vào xu hướng giảm - Downtrend), việc đa dạng hóa danh mục theo cách truyền thống (mua nhiều mã cổ phiếu khác nhau trong rổ VN30) trở nên **vô hiệu**. Biến số PC1 sẽ kích hoạt tâm lý đám đông và kéo toàn bộ các mã sụt giảm đồng loạt. Cách phòng vệ duy nhất trong giai đoạn này là chủ động hạ tỷ trọng đòn bẩy (Margin), đưa danh mục về tiền mặt hoặc sử dụng vị thế Short hợp đồng tương lai chỉ số VN30F để phòng hộ (Hedging).

### 2️⃣ Nhận diện Chu kỳ Luân chuyển Dòng tiền thông qua Nhân tố PC2 (Sector Rotation Tracking)

Hệ số tải trọng (Factor Loadings) của PC2 vạch trần một ranh giới trực giao độc lập về dòng tiền giữa hai chiến tuyến:

* **Cực Dương (Nhóm Năng lượng, Sản xuất & Bank Quốc doanh):** Thống trị bởi `GAS`, `PLX`, `VNM`, `VCB`, `BID`. Đây là nhóm sở hữu cấu trúc tài chính an toàn, dòng tiền mặt khỏe. Dòng tiền thông minh có xu hướng đổ vào cực Dương để **trú ẩn** khi thị trường chung có dấu hiệu rủi ro cao hoặc vĩ mô bất ổn.
* **Cực Âm (Nhóm Chu kỳ, Bất động sản & Bank Tư nhân):** Bị thống trị bởi nhóm cổ phiếu nhạy cảm cao với chu kỳ tín dụng như họ nhà Vin (`VHM`, `VIC`, `VRE`) và nhóm ngân hàng thương mại cổ phần tư nhân (`TCB`, `VPB`). Nhóm này bùng nổ khi nền kinh tế bước vào giai đoạn nới lỏng tiền tệ, thị trường tăng trưởng dựa trên động lực đầu cơ và khẩu vị chấp nhận rủi ro (Risk-on) của nhà đầu tư tăng cao.

Mô hình PCA đóng vai trò như một **la bàn định lượng** giúp nhà quản lý quỹ nhận diện sớm chân sóng của các nhóm ngành để thực hiện chiến lược xoay vòng danh mục (Sector Rotation) tối ưu.

### 3️⃣ Chiến lược Tối ưu hóa Cấu trúc Danh mục (Portfolio Diversification Strategy)

Từ cấu trúc phân hóa của PC2, quy tắc xương máu cho nhà đầu tư là **không bao giờ nắm giữ đồng thời hai cổ phiếu nằm cùng một phía của PC2** có hệ số loading tương đương (Ví dụ: đã mua `VHM` lại mua thêm `TCB`, hoặc đã mua `GAS` lại phân bổ thêm vào `PLX`). Sự trùng lặp này sẽ làm gia tăng mức độ tổn thương của danh mục khi dòng tiền rút khỏi phân khúc đó.

Chiến lược đi tiền thông minh là phối hợp bắt cặp chéo giữa các mã ở hai cực đối lập (Ví dụ: Kết hợp một mã thuộc nhóm Ngân hàng quốc doanh/Năng lượng ở cực Dương như `VCB`/`GAS` với một mã tài chính tư nhân chu kỳ ở cực Âm như `TCB`/`VHM`). Sự bù trừ và triệt tiêu biến động lẫn nhau giữa hai nhóm trực giao này giúp danh mục duy trì bộ đệm phòng vệ tự nhiên, giảm thiểu tối đa mức sụt giảm tài sản (Drawdown) trước mọi biến động đảo lớp của dòng tiền lớn trên thị trường.
