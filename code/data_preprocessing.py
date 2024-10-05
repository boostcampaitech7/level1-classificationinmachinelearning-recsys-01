import os
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

'''
# 결측치 처리
- train data에서 결측치가 100%인 컬럼은 학습할 수 없기 때문에 사용하지 않음
- 결측치가 있을 경우 이동평균(window_size=3)으로 처리
- 이동평균으로 대체되지 않는 결측치가 남아있는 경우 앞에 있는 값으로 결측치 처리
'''
def preprocess_missisng_value(df):
    # missing value check
    train_df = df.loc[df["_type"] == "train"]

    # 각 열에서 누락된 값의 수 & 백분율 계산
    missing_values = train_df.isnull().sum()
    missing_percentage = (missing_values / len(train_df)) * 100

    # 누락된 값 비율을 기준으로 열 정렬
    sorted_missing_percentage = missing_percentage.sort_values(ascending=False)

    # missing_value의 비율이 100%가 아닌 column만 추출
    non_missing_columns = sorted_missing_percentage[sorted_missing_percentage != 100.0].index.tolist()
    non_missing_columns.remove('ID')
    non_missing_columns.remove('target')
    non_missing_columns.remove('_type')

    # 결측치가 100%가 아닌 column만 사용
    new_data = train_df[['ID','target', '_type'] + non_missing_columns]

    # 이동평균으로 결측치 대체
    new_df_stab = new_data[non_missing_columns]

    # train 데이터에서 결측치 처리
    window_size = 3
    new_df_stab = new_df_stab.apply(lambda col: col.fillna(col.rolling(window=window_size, min_periods=1).mean()))
    new_df_stab = new_df_stab.fillna(method='ffill')


    # 결측치 처리한 new_df 정의
    new_train_df = pd.concat([new_data[['ID','target','_type']], new_df_stab], axis=1)

    # nan or inf 데이터 여부 확인
    for col in non_missing_columns:
        if (np.isnan(new_train_df[col]).any()) | (np.isinf(new_train_df[col]).any()):
            print(col)
            break

    # 테스트 데이터 결측치 처리
    test_df = df.loc[df["_type"] == "test"]

    # train 데이터에서 사용하는 컬럼만 가져옴
    new_test_df = test_df[['ID','target','_type'] + non_missing_columns]
    new_test_stab = new_test_df[non_missing_columns]

    # test
    window_size = 3
    new_test_stab = new_test_stab.apply(lambda col: col.fillna(col.rolling(window=window_size, min_periods=1).mean()))
    new_test_stab = new_test_stab.fillna(method='ffill')

    new_test_df = pd.concat([new_test_df[['ID','target','_type']], new_test_stab], axis=1)

    # train data에는 값이 있지만, test data에는 값이 없는 경우 컬럼 제거(종가 제외)
    # 결측치 비율을 계산
    missing_percentage = new_test_df.isnull().mean() * 100

    # 결측치 비율이 100%인 컬럼 이름만 출력
    columns_with_all_missing = missing_percentage[missing_percentage >= 50].index.tolist()

    # 100% 결측치가 있는 컬럼 제거
    '''
    - train data에는 값이 있어도, test data에는 값이 없는 경우에는 컬럼을 사용하지 않음
    - 하지만 종가(close)는 예측해야 하는 target과 연관성이 깊기 때문에 제거하지 않음
    '''
    columns_with_all_missing = [col for col in columns_with_all_missing if col not in ['target', 'hourly_market-data_price-ohlcv_all_exchange_spot_btc_usd_close']]
    new_train_df = new_train_df.drop(columns=columns_with_all_missing, errors='ignore')
    new_test_df = new_test_df.drop(columns=columns_with_all_missing, errors='ignore')

    return new_train_df, new_test_df

'''
# 이상치 처리
- 이동평균을 기준으로 이동평균의 표준편차에서 2배 이상 벗어나는 값을 이상치로 설정
- 이상치를 이동평균으로 대체
- 비트코인은 단기간의 변화에 민감하기 때문에 window_size는 3으로 짧게 설정
'''
def replace_outlier(df, window=3, threshold=2):
    df_cleaned = df.copy()
    
    # 숫자형 컬럼들에 대해 처리
    for column in df_cleaned.select_dtypes(include=[np.number]).columns:
        # 이동평균과 표준편차 계산
        rolling_mean = df_cleaned[column].rolling(window=window, min_periods=1).mean()
        rolling_std = df_cleaned[column].rolling(window=window, min_periods=1).std()

        # 이상치 기준 설정
        outliers = np.abs(df_cleaned[column] - rolling_mean) > (threshold * rolling_std)

        # 이상치를 이동평균으로 대체
        df_cleaned.loc[outliers, column] = rolling_mean[outliers]
    
    return df_cleaned

# Standardization으로 정규화
def standardization(train_df, test_df):
    features_to_scale = [col for col in train_df.columns if col not in ['ID', 'target', '_type']]

    scaler = StandardScaler()

    # 훈련 데이터 정규화
    train_df_scaled = train_df.copy()
    train_df_scaled[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])

    # 테스트 데이터 정규화
    test_df_scaled = test_df.copy()
    test_df_scaled[features_to_scale] = scaler.transform(test_df[features_to_scale])

    return train_df_scaled, test_df_scaled