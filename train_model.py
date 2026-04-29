import pandas as pd
import numpy as np
import os
import warnings
from datetime import timedelta
import subprocess
import sys

try:
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import shap
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "lightgbm", "optuna", "scikit-learn", "shap"])
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) 

data_path = 'data'
train_file = os.path.join(data_path, 'train_ready_dataset.csv')
sub_file = os.path.join('datathon-2026-round-1', 'sample_submission.csv')

print("1. Tải và quét lại Dữ liệu cực sạch mới...")
train_df = pd.read_csv(train_file)
train_df['date'] = pd.to_datetime(train_df['date'])
train_df = train_df.sort_values('date').reset_index(drop=True)

cpi_df = pd.read_csv('cpi.csv')
cpi_df['Thời Gian'] = pd.to_datetime(cpi_df['Thời Gian'])
cpi_df['year'] = cpi_df['Thời Gian'].dt.year
cpi_df['month'] = cpi_df['Thời Gian'].dt.month

def get_inflation_factor(date):
    match = cpi_df[(cpi_df['year'] == date.year) & (cpi_df['month'] == date.month)]
    if len(match) > 0:
        return match.iloc[0]['CPI'] / 100.0
    return cpi_df.iloc[-1]['CPI'] / 100.0

target_col = 'actual_revenue'
ignore_cols = ['date', target_col, 'inflation_factor', 'CPI']
features = [c for c in train_df.columns if c not in ignore_cols]

print(f"Sử dụng {len(features)} SOTA features tinh gọn!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import TimeSeriesSplit

n_samples = len(train_df)
c2 = int(0.8 * n_samples)

X_train_val = train_df.iloc[:c2][features]
y_train_val = train_df.iloc[:c2][target_col]
y_train_val_log = np.log1p(y_train_val)

X_test = train_df.iloc[c2:][features]
y_test = train_df.iloc[c2:][target_col]
y_test_log = np.log1p(y_test)

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

print("\n2. [OPTUNA] Tối ưu hóa mô hình với Time-Series Cross Validation & Early Stopping...")
def objective(trial):
    param_xgb = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }
    param_lgb = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 6),
        'num_leaves': trial.suggest_int('lgb_leaves', 20, 60),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    for train_index, val_index in tscv.split(X_train_val):
        X_tr, X_va = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_tr, y_va = y_train_val_log.iloc[train_index], y_train_val_log.iloc[val_index]
        
        m_xgb = xgb.XGBRegressor(**param_xgb, early_stopping_rounds=30)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        m_lgb = lgb.LGBMRegressor(**param_lgb)
        try:
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        except:
            m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=30, verbose=False)
            
        pred_xgb_log = m_xgb.predict(X_va)
        pred_lgb_log = m_lgb.predict(X_va)
        
        pred_ens_log = (pred_xgb_log + pred_lgb_log) / 2
        pred_ens = np.expm1(pred_ens_log)
        
        val_inflation = train_df.iloc[:c2]['inflation_factor'].iloc[val_index].values
        mae, rmse, r2 = eval_metrics(np.expm1(y_va) * val_inflation, pred_ens * val_inflation)
        cv_scores.append(rmse)
        
    return np.mean(cv_scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print("  => Best Params:", study.best_params)

best = study.best_params
final_xgb = xgb.XGBRegressor(
    n_estimators=best['xgb_n_estimators'], learning_rate=best['xgb_lr'], 
    max_depth=best['xgb_max_depth'], objective='reg:squarederror', n_jobs=-1, random_state=42)
final_lgb = lgb.LGBMRegressor(
    n_estimators=best['lgb_n_estimators'], learning_rate=best['lgb_lr'], 
    max_depth=best['lgb_max_depth'], num_leaves=best['lgb_leaves'], n_jobs=-1, random_state=42, verbose=-1)

final_xgb.fit(X_train_val, y_train_val_log)
final_lgb.fit(X_train_val, y_train_val_log)

print("\n=> Đánh giá trên tập TEST (20% cuối)...")
pred_xgb_test_log = final_xgb.predict(X_test)
pred_lgb_test_log = final_lgb.predict(X_test)
pred_ens_test = np.expm1((pred_xgb_test_log + pred_lgb_test_log) / 2)

test_inflation = train_df.iloc[c2:]['inflation_factor'].values
mae_test, rmse_test, r2_test = eval_metrics(y_test * test_inflation, pred_ens_test * test_inflation)
print(f"TEST MAE:  {mae_test:.4f}")
print(f"TEST RMSE: {rmse_test:.4f}")
print(f"TEST R²:   {r2_test:.4f}")

# --- THÊM PHẦN SHAP & FEATURE IMPORTANCE & PIPELINE (BÁO CÁO KỸ THUẬT) ---
print("\n=> [Báo Cáo Kỹ Thuật] 1. Feature Importance (Top 10 - XGBoost):")
feature_importances = pd.Series(final_xgb.feature_importances_, index=features).sort_values(ascending=False)
print(feature_importances.head(10))

print("\n=> [Báo Cáo Kỹ Thuật] 2. Tính toán SHAP values (Giải thích mô hình)...")
explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_test)
print(f"Tính SHAP values thành công! Kích thước: {shap_values.shape}.")
print("(Có thể dùng shap.summary_plot(shap_values, X_test) để trực quan hóa trong Notebook)")

print("\n=> [Báo Cáo Kỹ Thuật] 3. Tích hợp scikit-learn Pipeline (Chuẩn kỹ thuật MLOps)...")
# Khởi tạo pipeline mẫu cho LightGBM
pipeline_lgb = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('lgb', lgb.LGBMRegressor(
        n_estimators=best['lgb_n_estimators'], 
        learning_rate=best['lgb_lr'], 
        max_depth=best['lgb_max_depth'], 
        num_leaves=best['lgb_leaves'], 
        n_jobs=-1, random_state=42, verbose=-1))
])
pipeline_lgb.fit(X_train_val, y_train_val_log)
pipeline_score = pipeline_lgb.score(X_test, y_test_log)
print(f"Pipeline LGB Test R² (Log scale): {pipeline_score:.4f}")
print("Tích hợp Pipeline thành công!")
# ------------------------------------------------

# --- THÊM PHẦN PHÂN TÍCH LỖI (ERROR ANALYSIS) ---
print("\n=> Tiến hành Phân Tích Lỗi (Error Analysis) theo kinh nghiệm thực chiến...")
error_df = pd.DataFrame({
    'date': train_df.iloc[c2:]['date'],
    'actual': y_test * test_inflation,
    'pred': pred_ens_test * test_inflation,
    'abs_error': np.abs((y_test * test_inflation) - (pred_ens_test * test_inflation))
})
print("\n[ERROR ANALYSIS] Top 5 ngày mô hình dự báo SAI NHIỀU NHẤT (Worst 5):")
worst = error_df.sort_values('abs_error', ascending=False).head(5)
print(worst.to_string(index=False))

print("\n[ERROR ANALYSIS] Top 5 ngày mô hình dự báo CHUẨN XÁC NHẤT (Best 5):")
best = error_df.sort_values('abs_error', ascending=True).head(5)
print(best.to_string(index=False))
# ------------------------------------------------

# Train on all data for future inference
print("\n=> Retrain Ensemble model trên toàn bộ dữ liệu để dự báo tương lai...")
X_all = train_df[features]
y_all_log = np.log1p(train_df[target_col])
final_xgb.fit(X_all, y_all_log)
final_lgb.fit(X_all, y_all_log)
print("=> Train Ensemble model hoàn tất!")

# -------------------------------------------------------------
# [4] ITERATIVE INFERENCE NÂNG CAO TƯƠNG LẠI XA
# -------------------------------------------------------------
print("\n3. Đang nội suy tương lai (Inference) không đứt quãng...")
sample_sub = pd.read_csv(sub_file)
sample_sub['Date'] = pd.to_datetime(sample_sub['Date'])
future_dates = sample_sub.sort_values('Date')['Date'].tolist()

history_df = train_df.tail(60).copy()
try:
    import holidays
    vn_holidays = holidays.VN()
except:
    vn_holidays = []
    
promo_df = None
try:
    promo_path = os.path.join('datathon-2026-round-1', 'promotions.csv')
    promo_df = pd.read_csv(promo_path)
    promo_df['start_date'] = pd.to_datetime(promo_df['start_date'])
    promo_df['end_date'] = pd.to_datetime(promo_df['end_date'])
except:
    pass

predictions = []

for cur_date in future_dates:
    row_dict = {'date': cur_date}
    
    # Calendar & External
    row_dict['year'] = cur_date.year
    row_dict['month'] = cur_date.month
    row_dict['day'] = cur_date.day
    row_dict['dayofweek'] = cur_date.dayofweek
    row_dict['is_weekend'] = int(cur_date.dayofweek in [5, 6])
    row_dict['quarter'] = (cur_date.month - 1) // 3 + 1
    
    dayofyear = cur_date.timetuple().tm_yday
    row_dict['dayofyear'] = dayofyear
    row_dict['week'] = int(cur_date.isocalendar()[1])
    
    import math
    K = 2
    for k in range(1, K + 1):
        row_dict[f'sin_year_{k}'] = np.sin(2 * math.pi * k * dayofyear / 365.25)
        row_dict[f'cos_year_{k}'] = np.cos(2 * math.pi * k * dayofyear / 365.25)
        
    row_dict['is_holiday'] = 1 if cur_date in vn_holidays else 0
    row_dict['is_promo_active'] = 1 if (promo_df is not None and len(promo_df[(promo_df['start_date'] <= cur_date) & (promo_df['end_date'] >= cur_date)]) > 0) else 0
    row_dict['CPI'] = get_inflation_factor(cur_date) * 100.0
    
    # 💯 Tái cấu trúc Lags chuẩn chỉnh không bị Leakage hay Flat
    for f in features:
        if f == target_col + '_lag1':
            row_dict[f] = history_df.iloc[-1][target_col]
        elif f == target_col + '_lag7':
            row_dict[f] = history_df.iloc[-7][target_col] if len(history_df)>=7 else history_df.iloc[-1][target_col]
        elif f == target_col + '_lag14':
            row_dict[f] = history_df.iloc[-14][target_col] if len(history_df)>=14 else history_df.iloc[-1][target_col]
        elif f == target_col + '_lag30':
            row_dict[f] = history_df.iloc[-30][target_col] if len(history_df)>=30 else history_df.iloc[-1][target_col]
        
    # Tính Difference 1 - 7
    if 'revenue_diff_lag1_lag7' in features:
        row_dict['revenue_diff_lag1_lag7'] = row_dict.get(target_col + '_lag1', 0) - row_dict.get(target_col + '_lag7', 0)
        
    # Tính Dynamic Rolling & EWMA
    last_7_revs = history_df[target_col].tail(7)
    last_14_revs = history_df[target_col].tail(14)
    last_30_revs = history_df[target_col].tail(30)
    
    if 'revenue_roll_mean_7_lag1' in features: row_dict['revenue_roll_mean_7_lag1'] = last_7_revs.mean()
    if 'revenue_roll_std_7_lag1' in features: row_dict['revenue_roll_std_7_lag1'] = last_7_revs.std()
    if 'revenue_roll_mean_14_lag1' in features: row_dict['revenue_roll_mean_14_lag1'] = last_14_revs.mean()
    if 'revenue_roll_mean_30_lag1' in features: row_dict['revenue_roll_mean_30_lag1'] = last_30_revs.mean()
    
    if 'revenue_ewma_7_lag1' in features: row_dict['revenue_ewma_7_lag1'] = last_7_revs.ewm(span=7).mean().iloc[-1] if len(last_7_revs)>0 else 0
    if 'revenue_ewma_30_lag1' in features: row_dict['revenue_ewma_30_lag1'] = last_30_revs.ewm(span=30).mean().iloc[-1] if len(last_30_revs)>0 else 0
    
    # Điền giá trị chót dồn về lại cho covariate tĩnh (traffic smoothing)
    for f in features:
        if f not in row_dict:
            row_dict[f] = history_df.iloc[-1][f]
            
    cur_X = pd.DataFrame([row_dict])[features]
    
    pred_xgb_log = final_xgb.predict(cur_X)[0]
    pred_lgb_log = final_lgb.predict(cur_X)[0]
    
    pred_revenue = np.expm1((pred_xgb_log + pred_lgb_log) / 2.0)
    if pred_revenue < 0: pred_revenue = 0
    cur_inf = row_dict['CPI'] / 100.0
    predictions.append(pred_revenue * cur_inf)
    
    row_dict[target_col] = pred_revenue
    history_df = pd.concat([history_df, pd.DataFrame([row_dict])], ignore_index=True)

print(f"\n4. Đã nội suy liên tiếp {len(predictions)} ngày (tương lai khép kín).")
sample_sub['Revenue'] = predictions
output_file = os.path.join('datathon-2026-round-1', 'submission_sota_clean.csv')
sample_sub.to_csv(output_file, index=False)
print(f"🎉 Ghi xong kết quả ĐẶC ĐIỂM CHUẨN XÁC tại: {output_file}")
