import pandas as pd
import numpy as np
import os
import subprocess, sys

try:
    import holidays
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays"])
    import holidays

import warnings
warnings.filterwarnings('ignore')

data_path = 'data'
print("1. Reading sales data...")
sales_df = pd.read_csv(os.path.join(data_path, 'daily_sales_data_final.csv'))
sales_df['date'] = pd.to_datetime(sales_df['date'])
sales_df = sales_df.sort_values('date').reset_index(drop=True)

print("2. Tinh chỉnh Lịch & Ngày Lễ (Holidays)...")
sales_df['year'] = sales_df['date'].dt.year
sales_df['month'] = sales_df['date'].dt.month
sales_df['day'] = sales_df['date'].dt.day
sales_df['dayofweek'] = sales_df['date'].dt.dayofweek
sales_df['is_weekend'] = sales_df['dayofweek'].isin([5, 6]).astype(int)
sales_df['quarter'] = sales_df['date'].dt.quarter
sales_df['dayofyear'] = sales_df['date'].dt.dayofyear
sales_df['week'] = sales_df['date'].dt.isocalendar().week.astype(int)

# Thêm Fourier Terms để AI học được tính mùa vụ một cách mượt mà (Smooth Seasonality)
import math
K = 2
for k in range(1, K + 1):
    sales_df[f'sin_year_{k}'] = np.sin(2 * math.pi * k * sales_df['dayofyear'] / 365.25)
    sales_df[f'cos_year_{k}'] = np.cos(2 * math.pi * k * sales_df['dayofyear'] / 365.25)

vn_holidays = holidays.VN()
sales_df['is_holiday'] = sales_df['date'].apply(lambda x: 1 if x in vn_holidays else 0)

print("3. Khởi tạo Khuyến mãi...")
try:
    promo_path = os.path.join('datathon-2026-round-1', 'promotions.csv')
    promo_df = pd.read_csv(promo_path)
    promo_df['start_date'] = pd.to_datetime(promo_df['start_date'])
    promo_df['end_date'] = pd.to_datetime(promo_df['end_date'])
    
    def check_promo(date):
        active = promo_df[(promo_df['start_date'] <= date) & (promo_df['end_date'] >= date)]
        return 1 if len(active) > 0 else 0
        
    sales_df['is_promo_active'] = sales_df['date'].apply(check_promo)
except Exception as e:
    print("Warning: Could not process promotions:", e)
    sales_df['is_promo_active'] = 0

print("3.5 Áp dụng chuẩn hóa lạm phát (Inflation Normalization) bằng CPI...")
cpi_df = pd.read_csv('cpi.csv')
cpi_df['Thời Gian'] = pd.to_datetime(cpi_df['Thời Gian'])
cpi_df['year'] = cpi_df['Thời Gian'].dt.year
cpi_df['month'] = cpi_df['Thời Gian'].dt.month

sales_df = sales_df.merge(cpi_df[['year', 'month', 'CPI']], on=['year', 'month'], how='left')
sales_df['CPI'] = sales_df['CPI'].ffill().bfill()
sales_df['inflation_factor'] = sales_df['CPI'] / 100.0
sales_df['actual_revenue'] = sales_df['actual_revenue'] / sales_df['inflation_factor']

print("4. Sinh Features Lags dài hạn VÀ Bỏ các Lags gây đứt gãy tương lai...")
# Chỉ áp dụng lag cho mục tiêu để tự nuôi nhau, vứt bỏ order_id_lag1, gross_profit...
lag_features = ['actual_revenue']

for col in lag_features:
    sales_df[f'{col}_lag1'] = sales_df[col].shift(1)
    sales_df[f'{col}_lag7'] = sales_df[col].shift(7)
    sales_df[f'{col}_lag14'] = sales_df[col].shift(14)
    sales_df[f'{col}_lag30'] = sales_df[col].shift(30)

print("5. Sinh Gia tốc (Differential Features)...")
sales_df['revenue_diff_lag1_lag7'] = sales_df['actual_revenue_lag1'] - sales_df['actual_revenue_lag7']

print("6. Tái thiết lập Big Rolling & EWMA...")
sales_df['revenue_roll_mean_7_lag1'] = sales_df['actual_revenue'].shift(1).rolling(window=7).mean()
sales_df['revenue_roll_std_7_lag1']  = sales_df['actual_revenue'].shift(1).rolling(window=7).std()

sales_df['revenue_roll_mean_14_lag1'] = sales_df['actual_revenue'].shift(1).rolling(window=14).mean()
sales_df['revenue_roll_mean_30_lag1'] = sales_df['actual_revenue'].shift(1).rolling(window=30).mean()

sales_df['revenue_ewma_7_lag1'] = sales_df['actual_revenue'].shift(1).ewm(span=7).mean()
sales_df['revenue_ewma_30_lag1'] = sales_df['actual_revenue'].shift(1).ewm(span=30).mean()

# Loại bỏ gần như TẤT CẢ các Feature rác bị leakage khỏi training
leakage_cols = [c for c in sales_df.columns if 'revenue_category_' in c or 'revenue_segment' in c or 'revenue_color_' in c]
leakage_cols += ['gross_profit', 'order_id', 'quantity', 'customer_id', 'is_returned']

print("7. Trích chọn lại WebTraffic Rolling thay vì lag1...")
try:
    traffic_df = pd.read_csv(os.path.join(data_path, 'Fact_WebTraffic.csv'))
    date_col = next((c for c in traffic_df.columns if c.lower() == 'date'), None)
    if date_col:
        traffic_df['date'] = pd.to_datetime(traffic_df[date_col])
        if 'sessions' in traffic_df.columns:
            traffic_df['sessions_roll_30_lag1'] = traffic_df['sessions'].shift(1).rolling(30).mean()
            sales_df = sales_df.merge(traffic_df[['date', 'sessions_roll_30_lag1']], on='date', how='left')
except Exception:
    pass

final_df = sales_df.dropna()
final_train_cols = [c for c in final_df.columns if c not in leakage_cols]
train_ready_df = final_df[final_train_cols]

print("   -> SOTA Hình Hài Dữ liệu (Sạch bóng Cột rác):", train_ready_df.shape)
train_ready_df.to_csv(os.path.join(data_path, 'train_ready_dataset.csv'), index=False)
print("=> Đã ghi lại 'train_ready_dataset.csv'!")
