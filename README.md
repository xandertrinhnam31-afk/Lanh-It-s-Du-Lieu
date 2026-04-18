# DATATHON 2026 — The Gridbreakers
> Cuộc thi Khoa học Dữ liệu đầu tiên tại VinUniversity, tổ chức bởi **VinTelligence — VinUni DS&AI Club**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Thành viên nhóm

| # | Họ và tên | MSSV | Vai trò |
|---|-----------|------|---------|
| 1 | [Tên TM1] | [MSSV] | Data Engineer — Xử lý dữ liệu & Trắc nghiệm |
| 2 | [Tên TM2] | [MSSV] | Data Analyst — EDA Descriptive & Diagnostic |
| 3 | [Tên TM3] | [MSSV] | Business Analyst — EDA Predictive & Prescriptive |
| 4 | [Tên TM4] | [MSSV] | ML Engineer — Mô hình dự báo doanh thu |

---

## Tổng quan dự án

Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam (2012–2022), gồm 14 file CSV trải qua 4 lớp: Master, Transaction, Analytical, Operational.

Bài thi gồm 3 phần:

| Phần | Nội dung | Điểm |
|------|----------|------|
| 1 | Câu hỏi Trắc nghiệm (10 câu) | 20đ |
| 2 | Trực quan hoá & Phân tích EDA | 60đ |
| 3 | Mô hình Dự báo Doanh thu (Kaggle) | 20đ |

---

## Cấu trúc thư mục

```
datathon-2026-gridbreakers/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── .gitkeep              # Không lưu CSV — xem hướng dẫn tải bên dưới
│
├── notebooks/
│   ├── 01_data_loading_and_mcq.ipynb        # TM1: Load data + Trả lời trắc nghiệm
│   ├── 02_eda_descriptive_diagnostic.ipynb  # TM2: Phân tích mô tả & chẩn đoán
│   ├── 03_eda_predictive_prescriptive.ipynb # TM3: Dự báo sơ bộ & đề xuất hành động
│   └── 04_forecasting_model.ipynb           # TM4: Mô hình dự báo doanh thu
│
├── src/
│   └── utils.py              # Utility functions dùng chung cho cả nhóm
│
├── report/
│   ├── report.tex            # Báo cáo LaTeX (NeurIPS template)
│   ├── report.pdf            # Bản PDF xuất từ LaTeX
│   └── figures/              # Hình ảnh biểu đồ sử dụng trong báo cáo
│
└── submission/
    ├── submission.csv        # File nộp Kaggle (Date, Revenue, COGS)
    └── sample_submission.csv # File mẫu từ Kaggle
```

---

## Hướng dẫn cài đặt môi trường

### Yêu cầu
- Python 3.10+
- pip hoặc conda

### Cài đặt thư viện

```bash
git clone https://github.com/[username]/datathon-2026-gridbreakers.git
cd datathon-2026-gridbreakers
pip install -r requirements.txt
```

---

## Hướng dẫn tải dữ liệu

Dữ liệu không được lưu trong repo. Tải từ Kaggle theo các bước sau:

**Cách 1 — Tải thủ công:**
1. Truy cập: https://www.kaggle.com/competitions/datathon-2026-round-1/data
2. Tải toàn bộ file ZIP về máy
3. Giải nén vào thư mục `data/`

**Cách 2 — Kaggle API (khuyến nghị):**
```bash
# Cài Kaggle CLI (nếu chưa có)
pip install kaggle

# Đặt file kaggle.json vào ~/.kaggle/
# (Tải từ: kaggle.com → Account → API → Create New Token)

# Tải dữ liệu
kaggle competitions download -c datathon-2026-round-1 -p data/
cd data && unzip datathon-2026-round-1.zip
```

Sau khi tải xong, thư mục `data/` sẽ có cấu trúc:
```
data/
├── products.csv
├── customers.csv
├── promotions.csv
├── geography.csv
├── orders.csv
├── order_items.csv
├── payments.csv
├── shipments.csv
├── returns.csv
├── reviews.csv
├── sales.csv
├── sample_submission.csv
├── inventory.csv
└── web_traffic.csv
```

---

## Hướng dẫn chạy lại kết quả

Chạy các notebook theo **đúng thứ tự** sau:

```bash
# Bước 1 — Load dữ liệu và kiểm tra
jupyter notebook notebooks/01_data_loading_and_mcq.ipynb

# Bước 2 — EDA Descriptive & Diagnostic
jupyter notebook notebooks/02_eda_descriptive_diagnostic.ipynb

# Bước 3 — EDA Predictive & Prescriptive
jupyter notebook notebooks/03_eda_predictive_prescriptive.ipynb

# Bước 4 — Train mô hình và tạo submission
jupyter notebook notebooks/04_forecasting_model.ipynb
```

> **Lưu ý:** Tất cả notebook đã được đặt `random_seed = 42`. Kết quả phải tái lập hoàn toàn trên mọi máy có cùng dữ liệu và môi trường.

---

## Kết quả mô hình (Phần 3)

| Chỉ số | Giá trị (CV) | Giá trị (Kaggle Public) |
|--------|-------------|------------------------|
| MAE    | [...]       | [...]                  |
| RMSE   | [...]       | [...]                  |
| R²     | [...]       | [...]                  |

**Mô hình sử dụng:** [LightGBM / XGBoost / Prophet / Ensemble — điền sau]

**Top features (SHAP):**
1. [feature_1] — [giải thích ngắn]
2. [feature_2] — [giải thích ngắn]
3. [feature_3] — [giải thích ngắn]

---

## Đáp án Trắc nghiệm (Phần 1)

| Câu | Đáp án | Ghi chú |
|-----|--------|---------|
| Q1  | [A/B/C/D] | Median inter-order gap |
| Q2  | [A/B/C/D] | Gross margin theo segment |
| Q3  | [A/B/C/D] | Return reason — Streetwear |
| Q4  | [A/B/C/D] | Bounce rate thấp nhất |
| Q5  | [A/B/C/D] | % order_items có promo |
| Q6  | [A/B/C/D] | Age group đơn hàng cao nhất |
| Q7  | [A/B/C/D] | Region doanh thu cao nhất |
| Q8  | [A/B/C/D] | Payment method — cancelled |
| Q9  | [A/B/C/D] | Size return rate cao nhất |
| Q10 | [A/B/C/D] | Installment plan — giá trị cao nhất |

---

## Link nộp bài

- Kaggle submission: [link]
- Báo cáo PDF: [link hoặc xem thư mục /report]

---

## Ghi chú

- Toàn bộ đặc trưng mô hình được tạo **chỉ từ dữ liệu được cung cấp**, không sử dụng nguồn ngoài
- Cross-validation sử dụng `TimeSeriesSplit` để tránh data leakage theo chiều thời gian
- Mọi thắc mắc liên quan đến cuộc thi: liên hệ VinTelligence qua kênh chính thức
