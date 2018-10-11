# 中国高校计算机大赛
初赛 81/1392   复赛36/1392
precdict_active_user
本次比赛的内容请见https://www.kesci.com/home/competition/5ab8c36a8643e33f5138cba4

本次比赛是基于快手用户一个月内的注册、登录、拍摄行为四张表的数据来预测下一周用户是否活跃的赛题

processing.py 是复赛数据的做特征代码，读取文件的路径是在kesci复赛平台上存放数据的路径，复赛数据集无法下载，初赛数据可以在https://www.kesci.com/home/competition/5ab8c36a8643e33f5138cba4/content/4 下载

train.py 是调参的过程，选用用label的数据集1,2  其中1做训练集、2做验证集  寻找最优参数

predict.py 根据train.py选出的最优参数合并1，2两个数据集进行训练并对第三个提交线上评分的数据集进行预测
