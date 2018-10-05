# coding=utf-8

import lightgbm as lgb
import pandas as pd
import numpy as np

#feats = ['context_timestamp','context_id','predict_category_property','item_category_list','item_property_list','instance_id','click_feat_hour']

train1 = pd.read_csv('../input/train1.csv')
train2 = pd.read_csv('../input/train2.csv')



train = pd.concat([train2,train1])
del train1, train2
train_y = train.label
train_x = train.drop(['label','user_id'], axis=1)
del train
lgb_train = lgb.Dataset(train_x, train_y)
del train_x, train_y



test  = pd.read_csv('../input/train3.csv')


test_preds = test[['user_id']].copy()
test = test.drop('user_id', axis=1)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 120,
    'learning_rate': 0.01,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'lambda_l1': 2.0,
    'lambda_l2': 5.0,
    'min_gain_to_split': 0.01,
    'min_sum_hessian_in_leaf': 1.0,
    'max_depth': 7,
    'bagging_freq': 1
}







gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1200,
                categorical_feature=[ 'device_type','register_type'])



test_preds['predicted_score'] = gbm.predict(test)



print(test_preds.describe)
test_preds.to_csv("./lgb_submit.txt", index=None,header=None, sep=",")






