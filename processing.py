import pandas as pd
import numpy as np
import gc


def gen_train(feat_begin, feat_end, label, train_num, slide_day):
    user_register = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt', sep='\t',
                                names=['user_id', 'register_day', 'register_type', 'device_type'],
                                dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint16})

    user_register['register_type'] = user_register.register_type.apply(lambda x: str(x))
    user_register_type_dummy = pd.get_dummies(user_register.register_type)
    user_register_type_dummy.columns = ['register_type' + str(i) for i in range(user_register_type_dummy.shape[1])]
    user_register = pd.concat([user_register, user_register_type_dummy], axis=1)
    del user_register_type_dummy

    train = user_register[user_register.register_day <= feat_end]
    train.register_day = train.register_day.apply(lambda x: x - slide_day)
    current_day = 17
    train.register_day = train.register_day.apply(lambda x: min(current_day - x, 17))
    train.rename(columns={'register_day': 'register_day_distance'}, inplace=True)
    sum_user = train.shape[0]

    def register_day_weight(x):
        sum_weight = 0
        for i in range(x):
            sum_weight += day_to_weight(16 - i)
        return sum_weight

    def day_to_weight(x):
        if x == 16:
            return 0.5
        elif x >= 13:
            return 0.25
        elif x >= 10:
            return 0.125
        elif x >= 6:
            return 1 / float(16)
        else:
            return 1 / float(32)

    train['register_day_count'] = train.register_day_distance.apply(register_day_weight)

    train.drop('register_type', axis=1, inplace=True)
    print(train.info())

    app_launch = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt', sep='\t',
                             names=['user_id', 'day'],
                             dtype={0: np.uint32, 1: np.uint8})
    app_launch.drop_duplicates(inplace=True)
    train_app_launch = app_launch[(app_launch.day <= feat_end) & (app_launch.day >= feat_begin)]
    t11 = train_app_launch[['user_id']]
    t11['launch_times'] = 1
    t11 = t11.groupby('user_id').agg('sum').reset_index()

    user_id_latter = train_app_launch[(train_app_launch.day <= feat_end) & (train_app_launch.day >= feat_end)][
        ['user_id']]

    user_id_before = \
        train_app_launch[(train_app_launch.day <= feat_end - 7) & (train_app_launch.day >= feat_end - 7)][
            ['user_id']]

    cycle_user = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')
    for i in range(feat_end - 1, feat_end - 9 - 1, -1):
        user_id_latter = train_app_launch[(train_app_launch.day < i + 1) & (train_app_launch.day >= i)][['user_id']]

        user_id_before = train_app_launch[(train_app_launch.day < i - 7 + 1) & (train_app_launch.day >= i - 7)][
            ['user_id']]

        user_id_same = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')

        cycle_user.append(user_id_same)

    user_week = train_app_launch[(train_app_launch.day <= feat_end - 2) & (train_app_launch.day >= feat_end - 3)][
        ['user_id']]
    user_week.drop_duplicates(inplace=True)
    user_week_before = \
        train_app_launch[(train_app_launch.day <= feat_end - 2 - 7) & (train_app_launch.day >= feat_end - 3 - 7)][
            ['user_id']]
    user_week_before.drop_duplicates(inplace=True)
    user_week_both = pd.merge(user_week, user_week_before, on='user_id', how='inner')
    user_week_both['cycle_week_user'] = 1

    train_app_launch.day = train_app_launch.day.apply(lambda x: x - slide_day)
    train_app_launch['weight'] = train_app_launch.day.apply(register_day_weight)

    if label:
        label_user_id = app_launch[(app_launch.day <= feat_end + 7) & (app_launch.day >= feat_end + 1)][['user_id']]
        label_user_id.drop_duplicates(inplace=True)
    del app_launch
    gc.collect()

    t12 = train_app_launch.copy()
    t12 = t12.groupby('user_id')['day'].agg('max').reset_index()
    t12.day = t12.day.apply(lambda x: current_day - x)
    t12.rename(columns={'day': 'user_launch_last_day_distance'}, inplace=True)

    t13 = train_app_launch[['user_id', 'weight']]

    t13 = t13.groupby('user_id').agg('sum').reset_index()
    t13.rename(columns={'weight': 'user_launch_date_occupied'}, inplace=True)

    t11 = pd.merge(t11, t12, on='user_id', how='left')
    t11 = pd.merge(t11, t13, on='user_id', how='left')
    del t12, t13
    gc.collect()

    t = train_app_launch.copy()
    t = t.sort_values(by=['user_id', 'day'])
    t['diff_day'] = t.groupby('user_id')['day'].diff()

    t1 = t[['user_id', 'diff_day']]
    t2 = t1.groupby('user_id').agg('max').reset_index()
    t2.rename(columns={'diff_day': 'diff_day_max'}, inplace=True)
    t3 = t1.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'diff_day': 'diff_day_min'}, inplace=True)
    t4 = t1.groupby('user_id').agg('mean').reset_index()
    t4.rename(columns={'diff_day': 'diff_day_mean'}, inplace=True)

    t11 = pd.merge(t11, t2, on='user_id', how='left')
    t11 = pd.merge(t11, t3, on='user_id', how='left')
    t11 = pd.merge(t11, t4, on='user_id', how='left')

    del t, t1, t2, t3, t4

    for i in [2, 4, 7, 9]:
        t = train_app_launch[(train_app_launch.day < current_day) & (train_app_launch.day >= current_day - i)]
        tt = t[['user_id']]
        tt['%s%d_day_before_rate' % ('app_launch', i)] = 1
        tt = tt.groupby('user_id').agg('sum').reset_index()
        t11 = pd.merge(t11, tt, on='user_id', how='left')
        del tt

        tt = t[['user_id', 'weight']]
        tt = tt.groupby('user_id').agg('sum').reset_index()
        tt.rename(columns={'weight': '%s%d_day_before_occupy' % ('app_launch', i)}, inplace=True)
        gc.collect()

    train = pd.merge(train, t11, on='user_id', how='left')
    cycle_user.drop_duplicates(inplace=True)
    cycle_user['cycle_user'] = 1
    train = pd.merge(train, cycle_user, on='user_id', how='left')
    train = pd.merge(train, user_week_both, on='user_id', how='left')
    for i in [2, 4, 7, 9]:
        weight_ac = register_day_weight(i)
        train['weight_small'] = train.register_day_count.apply(lambda x: x if x < weight_ac else weight_ac)
        train['distance_small'] = train.register_day_distance.apply(lambda x: x if x < i else i)
        t['%s%d_day_before_rate' % ('app_launch', i)] = train['%s%d_day_before_rate' % (
        'app_launch', i)] / train.distance_small
        t['%s%d_day_before_occupy' % ('app_launch', i)] = train['%s%d_day_before_rate' % (
        'app_launch', i)] / train.weight_small
        train.drop(['weight_small', 'distance_small'], axis=1, inplace=True)

    train.diff_day_max.fillna(17, inplace=True)
    train.diff_day_min.fillna(17, inplace=True)
    train.diff_day_mean.fillna(17, inplace=True)
#    train['launch_times_avg'] = train.launch_times / train.register_day_distance
    train['user_launch_date_occupied_avg'] = train.user_launch_date_occupied / train.register_day_count
    train.drop(['launch_times', 'user_launch_date_occupied'], axis=1, inplace=True)
    print(train.info())

    video_create = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt', sep='\t',
                               names=['user_id', 'day'],
                               dtype={0: np.uint32, 1: np.uint8})
    t21 = video_create[(video_create.day <= feat_end) & (video_create.day >= feat_begin)]

    t22 = t21[['user_id']]
    t22['video_create_times'] = 1
    t22 = t22.groupby('user_id').agg('sum').reset_index()

    t = t21.copy()
    t.drop_duplicates(inplace=True)
    # user create video the day
    t = t.sort_values(by=['user_id', 'day'])
    t['create_video_diff_day'] = t.groupby('user_id')['day'].diff()

    t1 = t[['user_id', 'create_video_diff_day']]
    t2 = t1.groupby('user_id').agg('max').reset_index()
    t2.rename(columns={'create_video_diff_day': 'create_video_diff_day_max'}, inplace=True)
    t3 = t1.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'create_video_diff_day': 'create_video_diff_day_min'}, inplace=True)
    t4 = t1.groupby('user_id').agg('mean').reset_index()
    t4.rename(columns={'create_video_diff_day': 'create_video_diff_day_mean'}, inplace=True)

    t22 = pd.merge(t22, t2, on='user_id', how='left')
    t22 = pd.merge(t22, t3, on='user_id', how='left')
    t22 = pd.merge(t22, t4, on='user_id', how='left')
    del t, t1, t2, t3, t4

    user_id_latter = t21[(t21.day < feat_end + 1) & (t21.day >= feat_end)][['user_id']]
    user_id_latter.drop_duplicates(inplace=True)
    user_id_before = t21[(t21.day < feat_end - 7 + 1) & (t21.day >= feat_end - 7)][['user_id']]
    user_id_before.drop_duplicates(inplace=True)
    cycle_user_create_video = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')
    for i in range(feat_end - 1, feat_end - 9 - 1, -1):
        user_id_latter = t21[(t21.day < i + 1) & (t21.day >= i)][['user_id']]
        user_id_latter.drop_duplicates(inplace=True)
        user_id_before = t21[(t21.day < i - 7 + 1) & (t21.day >= i - 7)][['user_id']]
        user_id_before.drop_duplicates(inplace=True)
        user_id_same = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')

        cycle_user_create_video.append(user_id_same)

    user_week = t21[(t21.day <= feat_end - 2) & (t21.day >= feat_end - 3)][['user_id']]
    user_week.drop_duplicates(inplace=True)
    user_week_before = t21[(t21.day <= feat_end - 2 - 7) & (t21.day >= feat_end - 3 - 7)][['user_id']]
    user_week_before.drop_duplicates(inplace=True)
    user_week_both_create_video = pd.merge(user_week, user_week_before, on='user_id', how='inner')
    user_week_both_create_video['user_week_both_create_video'] = 1

    t21.day = t21.day.apply(lambda x: x - slide_day)
    t21['weight'] = t21.day.apply(day_to_weight)
    if label:
        label_user_id_video = video_create[(video_create.day <= feat_end + 7) & (video_create.day >= feat_end + 1)][
            ['user_id']]
        label_user_id.append(label_user_id_video)
        label_user_id.drop_duplicates(inplace=True)
    del video_create
    gc.collect()

    def weight_count(x):
        return x[0] * x[1]

    t23 = t21[['user_id', 'day']]
    t23['video_create_day_times'] = 1
    t23 = t23.groupby(['user_id', 'day']).agg('sum').reset_index()

    t24 = t23.groupby('user_id')['day'].agg('max').reset_index()
    t25 = pd.merge(t24, t23, on=['user_id', 'day'], how='left')
    t25.day = t25.day.apply(lambda x: current_day - x)
    t25.rename(columns={'day': 'user_video_create_latest_day_distance',
                        'video_create_day_times': 'user_video_create_latest_day_num'}, inplace=True)
    t22 = pd.merge(t22, t25, on='user_id', how='left')
    del t24, t25

    t27 = t23[['user_id', 'video_create_day_times']]
    t27 = t27.groupby('user_id').agg('max').reset_index()
    t27.rename(columns={'video_create_day_times': 'user_video_create_day_max'}, inplace=True)
    t22 = pd.merge(t22, t27, on='user_id', how='left')
    del t27
    t28 = t23[['user_id', 'video_create_day_times']]
    t28 = t28.groupby('user_id').agg('min').reset_index()
    t28.rename(columns={'video_create_day_times': 'user_video_create_day_min'}, inplace=True)
    t22 = pd.merge(t22, t28, on='user_id', how='left')
    del t28
    t29 = t23[['user_id', 'video_create_day_times']]
    t29 = t29.groupby('user_id').agg('mean').reset_index()
    t29.rename(columns={'video_create_day_times': 'user_video_create_day_mean'}, inplace=True)
    t22 = pd.merge(t22, t29, on='user_id', how='left')
    del t29

    t26 = t23[['user_id']]
    t26['day_num_create_video'] = 1
    t26 = t26.groupby('user_id').agg('sum').reset_index()
    t22 = pd.merge(t22, t26, on='user_id', how='left')
    del t26

    t201 = t21.copy()
    t201.drop_duplicates(inplace=True)
    t201 = t201[['user_id', 'weight']]
    t201 = t201.groupby('user_id').agg('sum').reset_index()
    t201.rename(columns={'weight': 'user_video_create_day_occupy'}, inplace=True)
    t22 = pd.merge(t22, t201, on='user_id', how='left')
    del t201

    t = t21[['user_id', 'weight']]
    t = t.groupby('user_id').agg('sum').reset_index()
    t.rename(columns={'weight': 'weight_create_count'}, inplace=True)
    t22 = pd.merge(t22, t, on='user_id', how='left')
    del t
    gc.collect()

    for i in [2, 4, 7, 9]:
        t = t21[(t21.day < current_day) & (t21.day >= current_day - i)]
        tt = t[['user_id']]
        del t
        tt['%s%d_day_before_sum' % ('video_create', i)] = 1
        tt = tt.groupby('user_id').agg('sum').reset_index()
        t22 = pd.merge(t22, tt, on='user_id', how='left')
        del tt
        gc.collect()

        t33 = t21[(t21.day < current_day) & (t21.day >= current_day - i)]
        t33['%s%d_day_sum_num' % ('video_create', i)] = 1
        t33 = t33.groupby(['user_id', 'day', 'weight']).agg('sum').reset_index()

        t = t21[(t21.day < current_day) & (t21.day >= current_day - i)][['user_id', 'weight']]
        t = t.groupby('user_id').agg('sum').reset_index()
        t.rename(columns={'weight': '%s%d_day_sum_weight' % ('video_create', i)}, inplace=True)
        t22 = pd.merge(t22, t, on='user_id', how='left')
        del t

        t = t33[['user_id', 'weight']]
        t = t.groupby('user_id').agg('sum').reset_index()
        t.rename(columns={'weight': '%s%d_day_occupy' % ('video_create', i)}, inplace=True)
        t22 = pd.merge(t22, t, on='user_id', how='left')
        del t

        t = t33[['user_id']]
        t['%s%d_num_day' % ('video_create', i)] = 1
        t = t.groupby('user_id').agg('sum').reset_index()
        t22 = pd.merge(t22, t, on='user_id', how='left')
        del t, t33

        gc.collect()

    train = pd.merge(train, t22, on='user_id', how='left')
    del t22
    cycle_user_create_video.drop_duplicates(inplace=True)
    cycle_user_create_video['cycle_user_create_video'] = 1
    train = pd.merge(train, cycle_user_create_video, on='user_id', how='left')
    train = pd.merge(train, user_week_both_create_video, on='user_id', how='left')

    for i in [2, 4, 7, 9]:
        weight_ac = register_day_weight(i)
        train['weight_activity'] = train.register_day_count.apply(lambda x: x if x < weight_ac else weight_ac)
        train['distance_activity'] = train.register_day_distance.apply(lambda x: x if x < i else i)
        train['%s%d_day_before_sum' % ('video_create', i)] = train['%s%d_day_before_sum' % (
        'video_create', i)] / train.distance_activity
        train['%s%d_day_sum_weight' % ('video_create', i)] = train['%s%d_day_sum_weight' % (
            'video_create', i)] / train.weight_activity
        train['%s%d_day_occupy' % ('video_create', i)] = train['%s%d_day_occupy' % (
            'video_create', i)] / train.weight_activity
        train['%s%d_num_day' % ('video_create', i)] = train['%s%d_num_day' % (
            'video_create', i)] / train.distance_activity
        train.drop(['distance_activity', 'weight_activity'], axis=1, inplace=True)

    gc.collect()
    train['video_create_avg'] = train.video_create_times / train.register_day_distance
    train['user_create_video_day_num_occupy'] = train.day_num_create_video / train.register_day_distance
    train['user_video_create_day_occupy_rate'] = train.user_video_create_day_occupy / train.register_day_count
    train['weight_create_count_rate'] = train.weight_create_count / train.register_day_count
    train.user_video_create_latest_day_distance.fillna(17, inplace=True)
    train.create_video_diff_day_max.fillna(17, inplace=True)
    train.create_video_diff_day_min.fillna(17, inplace=True)
    train.create_video_diff_day_mean.fillna(17, inplace=True)
    train.drop(['video_create_times', 'day_num_create_video', 'user_video_create_day_occupy', 'weight_create_count'],
               axis=1, inplace=True)
    print(train.info())

    user_activity = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt', sep='\t',
                                names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'],
                                dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint32, 4: np.uint32, 5: np.uint8})

    t31 = user_activity[(user_activity.day <= feat_end) & (user_activity.day >= feat_begin)]
    t32 = t31[['user_id']]
    t32['video_operate_times'] = 1
    t32 = t32.groupby('user_id').agg('sum').reset_index()

    user_id_latter = t31[(t31.day < feat_end + 1) & (t31.day >= feat_end)][['user_id']]
    user_id_latter.drop_duplicates(inplace=True)
    user_id_before = t31[(t31.day < feat_end - 7 + 1) & (t31.day >= feat_end - 7)][['user_id']]
    user_id_before.drop_duplicates(inplace=True)
    cycle_user_user_activity = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')
    for i in range(feat_end - 1, feat_end - 9 - 1, -1):
        user_id_latter = t31[(t31.day < i + 1) & (t31.day >= i)][['user_id']]
        user_id_latter.drop_duplicates(inplace=True)
        user_id_before = t31[(t31.day < i - 7 + 1) & (t31.day >= i - 7)][['user_id']]
        user_id_before.drop_duplicates(inplace=True)
        user_id_same = pd.merge(user_id_before, user_id_latter, on='user_id', how='inner')

        cycle_user_user_activity.append(user_id_same)

    user_week = t31[(t31.day <= feat_end - 2) & (t31.day >= feat_end - 3)][['user_id']]
    user_week.drop_duplicates(inplace=True)
    user_week_before = t31[(t31.day <= feat_end - 2 - 7) & (t31.day >= feat_end - 3 - 7)][['user_id']]
    user_week_before.drop_duplicates(inplace=True)
    user_week_both_user_activity = pd.merge(user_week, user_week_before, on='user_id', how='inner')
    user_week_both_user_activity['user_week_both_user_activity'] = 1

    t31.day = t31.day.apply(lambda x: x - slide_day)
    t31['weight'] = t31.day.apply(day_to_weight)
    if label:
        label_user_id_user_activity = \
            user_activity[(user_activity.day <= feat_end + 7) & (user_activity.day >= feat_end + 1)][['user_id']]
        label_user_id.append(label_user_id_user_activity)
        label_user_id.drop_duplicates(inplace=True)
    del user_activity
    gc.collect()

    '''
    对用户的行为进行加权


    def action_type_weight(lst):
        rst=0
        for i in lst:
            if i<=3:
                rst=rst+i+1
            else:
                rst=rst+i-6
        return rst

    t=t31[['user_id','action_type']]
    t=t.groupby('user_id')['action_type'].apply(action_type_weight)
    '''

    t33 = t31[['user_id', 'day', 'weight']]
    t33['user_operate_video_per_day'] = 1
    t33 = t33.groupby(['user_id', 'day', 'weight']).agg('sum').reset_index()

    t = t33.copy()
    t['user_operate_video_weight'] = t[['weight', 'user_operate_video_per_day']].apply(weight_count, axis=1)
    t = t[['user_id', 'user_operate_video_weight']]
    t = t.groupby('user_id').agg('sum').reset_index()
    t32 = pd.merge(t32, t, on='user_id', how='left')

    t34 = t33.groupby('user_id')['user_operate_video_per_day'].agg('max').reset_index()
    t34.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_max'}, inplace=True)
    t32 = pd.merge(t32, t34, on='user_id', how='left')
    del t34
    t35 = t33.groupby('user_id')['user_operate_video_per_day'].agg('min').reset_index()
    t35.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_min'}, inplace=True)
    t32 = pd.merge(t32, t35, on='user_id', how='left')
    del t35
    t36 = t33.groupby('user_id')['user_operate_video_per_day'].agg('mean').reset_index()
    t36.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_mean'}, inplace=True)
    t32 = pd.merge(t32, t36, on='user_id', how='left')
    del t36

    t36 = t33.groupby('user_id')['user_operate_video_per_day'].agg('median').reset_index()
    t36.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_median'}, inplace=True)
    t32 = pd.merge(t32, t36, on='user_id', how='left')
    del t36

    t36 = t33.groupby('user_id')['user_operate_video_per_day'].agg('std').reset_index()
    t36.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_std'}, inplace=True)
    t32 = pd.merge(t32, t36, on='user_id', how='left')
    del t36

    t36 = t33.groupby('user_id')['user_operate_video_per_day'].agg('var').reset_index()
    t36.rename(columns={'user_operate_video_per_day': 'user_operate_video_day_var'}, inplace=True)
    t32 = pd.merge(t32, t36, on='user_id', how='left')
    del t36

    t37 = t33[['user_id', 'day']]
    t38 = t37.groupby('user_id').agg('max').reset_index()
    t38 = pd.merge(t38, t37, on=['user_id', 'day'], how='left')
    del t37
    t38.day = t38.day.apply(lambda x: current_day - x)
    t38.rename(columns={'day': 'operate_video_distance_day'}, inplace=True)
    t32 = pd.merge(t32, t38, on='user_id', how='left')
    del t38

    t39 = t33[['user_id', 'weight']]
    t39 = t39.groupby('user_id').agg('sum').reset_index()
    t39.rename(columns={'weight': 'operate_video_day_occupy'}, inplace=True)
    t32 = pd.merge(t32, t39, on='user_id', how='left')
    del t39

    t301 = t33[['user_id']]
    del t33
    t301['user_operate_video_day_num'] = 1
    t301 = t301.groupby('user_id').agg('sum').reset_index()
    t32 = pd.merge(t32, t301, on='user_id', how='left')
    del t301

    # 下面处理one-hot后的单特征

    '''
    观看视频和author_id的习惯

    '''

    t = t31[['user_id', 'day', 'video_id', 'weight']]
    t.drop_duplicates(inplace=True)
    t = t[['user_id', 'video_id', 'weight']]
    t = t.groupby(['user_id', 'video_id']).agg('sum').reset_index()

    t2 = t.groupby('user_id')['weight'].agg('max').reset_index()
    t2.rename(columns={'weight': 'weight_user_view_video_weight'}, inplace=True)
    t32 = pd.merge(t32, t2, on='user_id', how='left')
    del t2, t

    t = t31[['user_id', 'day', 'author_id', 'weight']]
    t.drop_duplicates(inplace=True)
    t = t[['user_id', 'author_id', 'weight']]
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()

    t2 = t.groupby('user_id')['weight'].agg('max').reset_index()
    t2.rename(columns={'weight': 'user_view_same_author_weight'}, inplace=True)
    t32 = pd.merge(t32, t2, on='user_id', how='left')
    del t2, t

    for j in range(5):
        feature = 'page%d' % j
        t = t31[t31.page == j][['user_id', 'day', 'weight']]
        t[feature] = 1
        tt1 = t[['user_id', feature]]
        tt1 = tt1.groupby('user_id').agg('sum').reset_index()
        tt1.rename(columns={feature: 'user_operate_%s_sum' % feature}, inplace=True)
        t32 = pd.merge(t32, tt1, on='user_id', how='left')
        del tt1

        tt2 = t.groupby(['user_id', 'day', 'weight']).agg('sum').reset_index()
        del t

        tt6 = tt2[['user_id', 'day']]
        tt6 = tt6.groupby('user_id').agg('max').reset_index()
        tt6 = pd.merge(tt6, tt2, on=['user_id', 'day'], how='left')
        tt6.drop('weight', axis=1, inplace=True)
        tt6.day = tt6.day.apply(lambda x: current_day - x)
        tt6.rename(
            columns={feature: 'last_day_operate_%s_times' % feature, 'day': 'day_distance_to_operate_%s' % feature},
            inplace=True)
        t32 = pd.merge(t32, tt6, on='user_id', how='left')
        del tt6

        tt3 = tt2.groupby('user_id')[feature].agg('max').reset_index()
        tt3.rename(columns={feature: 'user_operated_%s_day_max' % feature}, inplace=True)
        t32 = pd.merge(t32, tt3, on='user_id', how='left')
        del tt3

        tt4 = tt2.groupby('user_id')[feature].agg('min').reset_index()
        tt4.rename(columns={feature: 'user_operated_%s_day_min' % feature}, inplace=True)
        tt4 = pd.merge(t32, tt4, on='user_id', how='left')
        del tt4

        tt5 = tt2.groupby('user_id')[feature].agg('median').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_median' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('std').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_std' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('var').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_var' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('mean').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_mean' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt7 = tt2[['user_id', 'weight']]
        tt7 = tt7.groupby('user_id').agg('sum').reset_index()
        tt7.rename(columns={'weight': 'user_operate_%s_occupy' % feature}, inplace=True)
        t32 = pd.merge(t32, tt7, on='user_id', how='left')
        del tt7

        tt8 = tt2[['user_id']]
        tt8['user_operate_%s_day_ocupy' % feature] = 1
        tt8 = tt8.groupby('user_id').agg('sum').reset_index()
        t32 = pd.merge(t32, tt8, on='user_id', how='left')
        del tt8
        del tt2
        gc.collect()

    for j in range(6):
        feature = 'action_type%d' % j
        t = t31[t31.action_type == j][['user_id', 'day', 'weight']]
        t[feature] = 1
        tt1 = t[['user_id', feature]]
        tt1 = tt1.groupby('user_id').agg('sum').reset_index()
        tt1.rename(columns={feature: 'user_operate_%s_sum' % feature}, inplace=True)
        t32 = pd.merge(t32, tt1, on='user_id', how='left')
        del tt1

        tt2 = t.groupby(['user_id', 'day', 'weight']).agg('sum').reset_index()
        del t

        tt6 = tt2[['user_id', 'day']]
        tt6 = tt6.groupby('user_id').agg('max').reset_index()
        tt6 = pd.merge(tt6, tt2, on=['user_id', 'day'], how='left')
        tt6.drop('weight', axis=1, inplace=True)
        tt6.day = tt6.day.apply(lambda x: current_day - x)
        tt6.rename(
            columns={feature: 'last_day_operate_%s_times' % feature, 'day': 'day_distance_to_operate_%s' % feature},
            inplace=True)
        t32 = pd.merge(t32, tt6, on='user_id', how='left')
        del tt6

        tt3 = tt2.groupby('user_id')[feature].agg('max').reset_index()
        tt3.rename(columns={feature: 'user_operated_%s_day_max' % feature}, inplace=True)
        t32 = pd.merge(t32, tt3, on='user_id', how='left')
        del tt3

        tt4 = tt2.groupby('user_id')[feature].agg('min').reset_index()
        tt4.rename(columns={feature: 'user_operated_%s_day_min' % feature}, inplace=True)
        tt4 = pd.merge(t32, tt4, on='user_id', how='left')
        del tt4

        tt5 = tt2.groupby('user_id')[feature].agg('median').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_median' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('std').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_std' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('var').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_var' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt5 = tt2.groupby('user_id')[feature].agg('mean').reset_index()
        tt5.rename(columns={feature: 'user_operated_%s_day_mean' % feature}, inplace=True)
        t32 = pd.merge(t32, tt5, on='user_id', how='left')
        del tt5

        tt7 = tt2[['user_id', 'weight']]
        tt7 = tt7.groupby('user_id').agg('sum').reset_index()
        tt7.rename(columns={'weight': 'user_operate_%s_occupy' % feature}, inplace=True)
        t32 = pd.merge(t32, tt7, on='user_id', how='left')
        del tt7

        tt8 = tt2[['user_id']]
        tt8['user_operate_%s_day_ocupy' % feature] = 1
        tt8 = tt8.groupby('user_id').agg('sum').reset_index()
        t32 = pd.merge(t32, tt8, on='user_id', how='left')
        del tt8
        del tt2
        gc.collect()

    for i in [2, 4, 7, 9]:
        t = t31[(t31.day < current_day) & (t31.day >= current_day - i)][['user_id', 'day', 'video_id', 'weight']]
        t.drop_duplicates(inplace=True)
        t = t[['user_id', 'video_id', 'weight']]
        t = t.groupby(['user_id', 'video_id']).agg('sum').reset_index()

        t2 = t.groupby('user_id')['weight'].agg('max').reset_index()
        t2.rename(columns={'weight': 'weight_user_view_video_%d_weight' % i}, inplace=True)
        t32 = pd.merge(t32, t2, on='user_id', how='left')
        del t2, t

        t = t31[(t31.day < current_day) & (t31.day >= current_day - i)][['user_id', 'day', 'author_id', 'weight']]
        t.drop_duplicates(inplace=True)
        t = t[['user_id', 'author_id', 'weight']]
        t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()

        t2 = t.groupby('user_id')['weight'].agg('max').reset_index()
        t2.rename(columns={'weight': 'user_view_same_author_%d_weight' % i}, inplace=True)
        t32 = pd.merge(t32, t2, on='user_id', how='left')
        del t2, t

        t = t31[(t31.day < current_day) & (t31.day >= current_day - i)]
        tt = t[['user_id']]
        del t
        tt['%s%d_day_sum' % ('user_activity', i)] = 1
        tt = tt.groupby('user_id').agg('sum').reset_index()
        t32 = pd.merge(t32, tt, on='user_id', how='left')
        del tt

        t33 = t31[(t31.day < current_day) & (t31.day >= current_day - i)][['user_id', 'day', 'weight']]
        t33['%s%d_day_sum_num' % ('user_activity', i)] = 1
        t33 = t33.groupby(['user_id', 'day', 'weight']).agg('sum').reset_index()

        t = t33.copy()
        t['%s%d_day_sum_weight' % ('user_activity', i)] = t[
            ['weight', '%s%d_day_sum_num' % ('user_activity', i)]].apply(
            weight_count, axis=1)
        t = t[['user_id', '%s%d_day_sum_weight' % ('user_activity', i)]]
        t = t.groupby('user_id').agg('sum').reset_index()
        t32 = pd.merge(t32, t, on='user_id', how='left')
        del t

        t = t33[['user_id', 'weight']]
        t = t.groupby('user_id').agg('sum').reset_index()
        t.rename(columns={'weight': '%s%d_day_occupy' % ('user_activity', i)}, inplace=True)
        t32 = pd.merge(t32, t, on='user_id', how='left')
        del t

        t = t33[['user_id']]
        t['%s%d_num_day' % ('user_activity', i)] = 1
        t = t.groupby('user_id').agg('sum').reset_index()
        t32 = pd.merge(t32, t, on='user_id', how='left')
        del t, t33

        gc.collect()

    t = t31[['user_id', 'day']]

    t = t.sort_values(by=['user_id', 'day'])
    t['user_activity_diff_day'] = t.groupby('user_id')['day'].diff()

    t1 = t[['user_id', 'user_activity_diff_day']]
    t2 = t1.groupby('user_id').agg('max').reset_index()
    t2.rename(columns={'user_activity_diff_day': 'user_activity_diff_day_max'}, inplace=True)
    t3 = t1.groupby('user_id').agg('min').reset_index()
    t3.rename(columns={'user_activity_diff_day': 'user_activity_diff_day_min'}, inplace=True)
    t4 = t1.groupby('user_id').agg('mean').reset_index()

    t4.rename(columns={'user_activity_diff_day': 'user_activity_diff_day_mean'}, inplace=True)
    t32 = pd.merge(t32, t2, on='user_id', how='left')
    t32 = pd.merge(t32, t3, on='user_id', how='left')
    t32 = pd.merge(t32, t4, on='user_id', how='left')
    del t, t1, t2, t3, t4

    '''
    author_id对用户的吸引力
    '''

    t = t31[['author_id']]
    t['author_viewed_num'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t['author_rank'] = t.author_viewed_num.rank(method="first", ascending=False)
    t = t[t.author_rank <= 30]
    author_list = list(t.author_id)
    del t

    t = t31[['user_id', 'author_id']]
    t.drop_duplicates(inplace=True)

    def get_unique(x):
        unique_data = set(list(x))
        return list(unique_data)

    t = t.groupby('user_id')['author_id'].apply(lambda x: get_unique(x)).reset_index()
    t.columns = ['user_id', 'user_view_author_list']

    t32 = pd.merge(t32, t, on='user_id', how='left')
    del t

    def attract_force(user_list, author_list):
        all = 0
        for i in user_list:
            if i in author_list:
                all += 1
        return all / float(30)

    t32['author_atttractive_force'] = t32.user_view_author_list.apply(lambda x: attract_force(x, author_list))

    train = pd.merge(train, t32, on='user_id', how='left')

    for feature in ['page0', 'page1', 'page2', 'page3', 'page4', 'action_type0',
                    'action_type1', 'action_type2', 'action_type3', 'action_type4',
                    'action_type5']:
#        train['user_operate_%s_rate' % feature] = train['user_operate_%s_sum' % feature] / train.video_operate_times
        train['user_operate_%s_sum' % feature] = train['user_operate_%s_sum' % feature] / train.register_day_distance
        train['day_distance_to_operate_%s' % feature] = train['day_distance_to_operate_%s' % feature].fillna(17)
        train['user_operate_%s_day_ocupy' % feature] = train[
                                                           'user_operate_%s_day_ocupy' % feature] / train.register_day_distance
        train['user_operate_%s_occupy' % feature] = train['user_operate_%s_occupy' % feature] / train.register_day_count
    train.operate_video_day_occupy = train.operate_video_day_occupy / train.register_day_count

    train.video_operate_times = train.video_operate_times / train.register_day_distance

    train.user_operate_video_day_num = train.user_operate_video_day_num / train.register_day_distance

    for i in [2, 4, 7, 9]:
        weight_ac = register_day_weight(i)
        train['weight_activity'] = train.register_day_count.apply(lambda x: x if x < weight_ac else weight_ac)
        train['distance_activity'] = train.register_day_distance.apply(lambda x: x if x < i else i)
        train['weight_user_view_video_%d_weight_rete' % i] = train[
                                                                 'weight_user_view_video_%d_weight' % i] / train.weight_activity
        train['user_view_same_author_%d_weight_rate' % i] = train[
                                                                'user_view_same_author_%d_weight' % i] / train.weight_activity
        train['%s%d_day_sum' % ('user_activity', i)] = train['%s%d_day_sum' % (
            'user_activity', i)] / train.distance_activity
        train['%s%d_day_sum_weight' % ('user_activity', i)] = train['%s%d_day_sum_weight' % (
            'user_activity', i)] / train.weight_activity
        train['%s%d_day_occupy' % ('user_activity', i)] = train['%s%d_day_occupy' % (
            'user_activity', i)] / train.weight_activity
        train['%s%d_num_day' % ('user_activity', i)] = train['%s%d_num_day' % (
            'user_activity', i)] / train.distance_activity
        train.drop(['distance_activity', 'weight_activity'], axis=1, inplace=True)

    t = t31[['author_id']]
    t = t.drop_duplicates()
    t['user_is_author'] = 1
    del t31
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    train = pd.merge(train, t, on='user_id', how='left')
    del t

    cycle_user_user_activity.drop_duplicates(inplace=True)
    cycle_user_user_activity['cycle_user_user_activity'] = 1
    train = pd.merge(train, cycle_user_user_activity, on='user_id', how='left')
    train = pd.merge(train, user_week_both_user_activity, on='user_id', how='left')

    train.operate_video_distance_day.fillna(17, inplace=True)
    train.user_activity_diff_day_max.fillna(17, inplace=True)
    train.user_activity_diff_day_min.fillna(17, inplace=True)
    train.user_activity_diff_day_mean.fillna(17, inplace=True)
    if label:
        label_user_id['label'] = 1
        train = pd.merge(train, label_user_id, on='user_id', how='left')
    train = train.drop(['register_day_count','user_view_author_list'], axis=1)

    train = train.fillna(0)
    print(train.info())

    train.to_csv('../input/v16_train%d.csv' % train_num, index=None)
    del train


gen_train(1, 16, True,  1,0)
gen_train(8, 23, True ,2,7)
gen_train(15, 30, False, 3, 14)
