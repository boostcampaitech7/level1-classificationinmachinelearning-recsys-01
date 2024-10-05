import os
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

'''
# close(종가) 예측
'''
def predict_close_price(std_train_df, std_test_df):
    # 타겟과 피처 설정
    y_train = std_train_df['hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close']
    X_train = std_train_df.drop(columns=['hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close', 'ID', 'target', '_type'], errors='ignore')

    # 모델 훈련
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # test_df에서 예측
    X_test = std_test_df.drop(columns=['hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close', 'ID', 'target', '_type'], errors='ignore')
    y_pred = model.predict(X_test)

    # new_test_df에 y_pred 값을 추가
    std_test_df['hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close'] = y_pred

    df = pd.concat([std_train_df, std_test_df], ignore_index=True)

    return df


def select_and_create_feature(df):
    # 모델에 사용할 컬럼, 컬럼의 rename rule을 미리 할당함
    cols_dict: Dict[str, str] = {
        "ID": "ID",
        "target": "target",
        "_type": "_type",
        "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
        "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
        "hourly_market-data_taker-buy-sell-stats_binance_taker_buy_ratio" : "buy_ratio_bi",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
        "hourly_market-data_taker-buy-sell-stats_binance_taker_sell_ratio" : "sell_ratio_bi",
        "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
        "hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close" : "close",
    }
    df=df[cols_dict.keys()].rename(cols_dict, axis=1)

    # 새로운 피처 생성
    df = df.assign(
        liquidation_diff=df["long_liquidations"] - df["short_liquidations"],
        volume_diff=df["buy_volume"] - df["sell_volume"],
        buy_sell_volume_ratio=df["buy_volume"] / (df["sell_volume"] + 1),
        close_diff = df['close'].diff().fillna(method='bfill'),
        volume = df["buy_volume"] + df["sell_volume"],
        SMA5 = df['close'].rolling(window=5).mean().fillna(method='bfill'),  
        SMA10 = df['close'].rolling(window=10).mean().fillna(method='bfill'),
        EMA5 = df['close'].ewm(span=5, adjust=False).mean().fillna(method='bfill'),  
        EMA3 = df['close'].ewm(span=10, adjust=False).mean().fillna(method='bfill'),
        close_trend5 = df['close'].pct_change(periods=5).fillna(method='bfill'),
        close_trend3 = df['close'].pct_change(periods=10).fillna(method='bfill'),
    )
    # category, continuous 열을 따로 할당해둠
    conti_cols: List[str] = [_ for _ in cols_dict.values() if _ not in ["ID", "target", "_type"]] + [
        "liquidation_diff",
        "volume_diff",
        "buy_sell_volume_ratio",
        "close_diff",
        "volume",
        "SMA10",
        "EMA3",
        "SMA5",
        "EMA5",
        "close_trend3",
        "close_trend5"
    ]

    return df, conti_cols

def shift_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> List[pd.Series]:
    """
    연속형 변수의 shift feature 생성
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        List[pd.Series]
    """
    df_shift_dict = [
        df[conti_col].shift(interval).rename(f"{conti_col}{interval}")
        for conti_col in conti_cols
        for interval in intervals
    ]
    return df_shift_dict