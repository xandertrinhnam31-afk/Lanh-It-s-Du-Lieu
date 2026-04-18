"""
utils.py — Utility functions dùng chung cho toàn nhóm
DATATHON 2026 — The Gridbreakers
"""

import pandas as pd
import numpy as np
import os

# ─── Đường dẫn dữ liệu ────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def data_path(filename: str) -> str:
    """Trả về đường dẫn đầy đủ tới file dữ liệu."""
    return os.path.join(DATA_DIR, filename)


# ─── Load dữ liệu ─────────────────────────────────────────────────────────────

def load_all_data() -> dict:
    """
    Load toàn bộ 14 file CSV vào dict.
    Trả về: {"products": df, "customers": df, ...}
    """
    files = {
        "products":    "products.csv",
        "customers":   "customers.csv",
        "promotions":  "promotions.csv",
        "geography":   "geography.csv",
        "orders":      "orders.csv",
        "order_items": "order_items.csv",
        "payments":    "payments.csv",
        "shipments":   "shipments.csv",
        "returns":     "returns.csv",
        "reviews":     "reviews.csv",
        "sales":       "sales.csv",
        "inventory":   "inventory.csv",
        "web_traffic": "web_traffic.csv",
        "sample_submission": "sample_submission.csv",
    }

    dfs = {}
    for key, fname in files.items():
        path = data_path(fname)
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
            print(f"  Loaded {fname}: {dfs[key].shape}")
        else:
            print(f"  [WARNING] File not found: {fname}")
    return dfs


# ─── Parse ngày tháng ─────────────────────────────────────────────────────────

DATE_COLS = {
    "orders":      ["order_date"],
    "customers":   ["signup_date"],
    "promotions":  ["start_date", "end_date"],
    "shipments":   ["ship_date", "delivery_date"],
    "returns":     ["return_date"],
    "reviews":     ["review_date"],
    "sales":       ["Date"],
    "inventory":   ["snapshot_date"],
    "web_traffic": ["date"],
}

def parse_dates(dfs: dict) -> dict:
    """Chuyển các cột ngày tháng sang kiểu datetime."""
    for key, cols in DATE_COLS.items():
        if key in dfs:
            for col in cols:
                if col in dfs[key].columns:
                    dfs[key][col] = pd.to_datetime(dfs[key][col])
    return dfs


# ─── Kiểm tra dữ liệu ─────────────────────────────────────────────────────────

def check_nulls(dfs: dict) -> pd.DataFrame:
    """Tóm tắt số lượng null của tất cả các bảng."""
    records = []
    for name, df in dfs.items():
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null > 0:
                records.append({
                    "table": name,
                    "column": col,
                    "null_count": n_null,
                    "null_pct": round(n_null / len(df) * 100, 2),
                })
    return pd.DataFrame(records).sort_values("null_pct", ascending=False)


def check_duplicates(dfs: dict) -> pd.DataFrame:
    """Kiểm tra số dòng duplicate trong từng bảng."""
    records = []
    for name, df in dfs.items():
        n_dup = df.duplicated().sum()
        records.append({"table": name, "rows": len(df), "duplicates": n_dup})
    return pd.DataFrame(records)


# ─── Gross Margin ─────────────────────────────────────────────────────────────

def add_gross_margin(products: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột gross_margin = (price - cogs) / price."""
    df = products.copy()
    df["gross_margin"] = (df["price"] - df["cogs"]) / df["price"]
    return df


# ─── Join helpers ─────────────────────────────────────────────────────────────

def orders_with_geography(orders: pd.DataFrame, geography: pd.DataFrame) -> pd.DataFrame:
    """Join orders với geography để lấy thông tin vùng."""
    return orders.merge(geography[["zip", "city", "region", "district"]], on="zip", how="left")


def order_items_with_products(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join order_items với products để lấy thông tin sản phẩm."""
    return order_items.merge(products, on="product_id", how="left")


def returns_with_products(returns: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join returns với products."""
    return returns.merge(products[["product_id", "product_name", "category", "segment", "size"]], 
                         on="product_id", how="left")


# ─── Revenue helpers ──────────────────────────────────────────────────────────

def compute_return_rate(returns: pd.DataFrame, order_items: pd.DataFrame,
                        products: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Tính return rate theo một cột nhóm (vd: 'size', 'category', 'segment').
    Return rate = số lượt return / số dòng order_items cho nhóm đó.
    """
    oi = order_items_with_products(order_items, products)
    ret = returns_with_products(returns, products)

    oi_count = oi.groupby(group_col).size().rename("order_item_count")
    ret_count = ret.groupby(group_col).size().rename("return_count")

    result = pd.concat([oi_count, ret_count], axis=1).fillna(0)
    result["return_rate"] = result["return_count"] / result["order_item_count"]
    return result.sort_values("return_rate", ascending=False)


# ─── Time series helpers ──────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Thêm các đặc trưng lịch: year, month, quarter, day_of_week, is_weekend."""
    df = df.copy()
    dt = df[date_col]
    df["year"]        = dt.dt.year
    df["month"]       = dt.dt.month
    df["quarter"]     = dt.dt.quarter
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon, 6=Sun
    df["is_weekend"]  = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "Revenue",
                     lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """Thêm lag features cho cột target. Đảm bảo df đã sort theo ngày."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "Revenue",
                          windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Thêm rolling mean và rolling std."""
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"]  = df[target_col].shift(1).rolling(w).std()
    return df


# ─── Seed ─────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Đặt random seed để đảm bảo tính tái lập."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    print(f"[seed] Random seed set to {seed}")
