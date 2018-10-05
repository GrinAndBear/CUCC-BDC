#coding=utf-8

import lightgbm as lgb
import pandas as pd
import numpy as np



train1 = pd.read_csv('../input/train1.csv')
train2 = pd.read_csv('../input/train2.csv')


print(train1.shape,train2.shape)

train1_y = train1.label
train1_x = train1.drop(['label','user_id'],axis=1)
del train1
lgb_train1 = lgb.Dataset(train1_x,train1_y)
del train1_x,train1_y

train2_y = train2.label
train2_x = train2.drop(['label','user_id'],axis=1)
del train2
lgb_train2 = lgb.Dataset(train2_x,train2_y)
del train2_x,train2_y


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 30,
    'learning_rate': 0.1,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'lambda_l1': 2.0,
    'lambda_l2': 5.0,
    'min_gain_to_split': 0.001,
    'min_sum_hessian_in_leaf': 1.0,
    'max_depth': 5,
    'bagging_freq': 1
}


gbm = lgb.train(params,
                lgb_train1,
                num_boost_round=20000,
                valid_sets=[lgb_train1, lgb_train2],
                early_stopping_rounds=100,
                categorical_feature=['register_type','device_type'])
                                     


# save feature score
fs = pd.DataFrame(columns=['feature','score'])
fs['feature'] = list(gbm.feature_name())
fs['score'] = list(gbm.feature_importance())
fs.sort_values(by='score',ascending=False,inplace=True)
fs.to_csv('../input/lgb_feature_score.csv',index=None)


