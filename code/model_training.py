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
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
# 모델 훈련
- LigthGBM, XGBoost, CatBoost, SVM 모델을 사용하여 스태킹 앙상블
- 메타 모델은 Logistic Regression model 사용
'''
def model_training(train_df, test_df):
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_df.drop(["target", "ID"], axis=1),
        train_df["target"].astype(int),
        test_size=0.2,
        random_state=42
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier()
    lgb_model.fit(x_train, y_train, eval_set=(x_valid, y_valid))

    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10, verbose=False)

    # CatBoost
    cat_model = CatBoostClassifier()
    cat_model.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=False)

    # SVM
    svm_model = SVC(probability=True)
    svm_model.fit(x_train, y_train)

    lgb_train_pred = lgb_model.predict_proba(x_train)
    xgb_train_pred = xgb_model.predict_proba(x_train)
    cat_train_pred = cat_model.predict_proba(x_train)
    svm_train_pred = svm_model.predict_proba(x_train)

    stacked_train = np.hstack((lgb_train_pred, xgb_train_pred, cat_train_pred, svm_train_pred))

    # meta model - logistic regression
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    meta_model.fit(stacked_train, y_train)

    lgb_valid_pred = lgb_model.predict_proba(x_valid)
    xgb_valid_pred = xgb_model.predict_proba(x_valid)
    cat_valid_pred = cat_model.predict_proba(x_valid)
    svm_valid_pred = svm_model.predict_proba(x_valid)

    stacked_valid = np.hstack((lgb_valid_pred, xgb_valid_pred, cat_valid_pred, svm_valid_pred))

    # 검증 세트에 대한 예측 및 평가
    y_valid_pred_class = meta_model.predict(stacked_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred_class)
    auroc = roc_auc_score(y_valid, meta_model.predict_proba(stacked_valid), multi_class="ovr")

    print(f"Stacking Ensemble Accuracy: {accuracy}, AUROC: {auroc}")

    # 전체 데이터를 이용해서 훈련
    lgb_test_pred = lgb_model.predict_proba(test_df.drop(["target","ID"], axis=1))
    xgb_test_pred = xgb_model.predict_proba(test_df.drop(["target","ID"], axis=1))
    cat_test_pred = cat_model.predict_proba(test_df.drop(["target", "ID"], axis=1))
    svm_test_pred = svm_model.predict_proba(test_df.drop(["target", "ID"], axis=1))

    stacked_test = np.hstack((lgb_test_pred, xgb_test_pred, cat_test_pred, svm_test_pred))

    # 메타 모델을 사용하여 최종 클래스 예측
    y_test_pred_class = meta_model.predict(stacked_test)

    return y_test_pred_class