import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import data_load
import data_preprocessing
import feature_engineering
import model_training

# 데이터 로드
submission_df, df = data_load.data_load()

# 데이터 전처리
train_df, test_df = data_preprocessing.preprocess_missisng_value(df)
cleaned_train_df = data_preprocessing.replace_outlier(train_df)
cleaned_test_df = data_preprocessing.replace_outlier(test_df)
std_train_df, std_test_df = data_preprocessing.standardization(cleaned_train_df, cleaned_test_df)

# feature_engineering
df = feature_engineering.predict_close_price(std_train_df, std_test_df)
df, conti_cols = feature_engineering.select_and_create_feature(df)

# 최대 24시간의 shift 피쳐를 계산
shift_list = feature_engineering.shift_feature(
    df=df, conti_cols=conti_cols, intervals=[ _ for _ in range (1, 24)]
)

# concat 하여 df 에 할당
df = pd.concat([df, pd.concat(shift_list, axis=1)], axis=1)

# 타겟 변수를 제외한 변수를 forwardfill, -999로 결측치 대체
_target = df["target"]
df = df.ffill().fillna(-999).assign(target = _target)

# _type에 따라 train, test 분리
train_df = df.loc[df["_type"]=="train"].drop(columns=["_type"])
test_df = df.loc[df["_type"]=="test"].drop(columns=["_type"])

# model training
y_test_pred_class = model_training.model_training(train_df, test_df) 
submission_df = submission_df.assign(target=y_test_pred_class)
submission_df.to_csv("output_last.csv", index=False)
