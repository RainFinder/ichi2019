# coding=utf-8
from sklearn.externals import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
import os
from sklearn.model_selection import train_test_split
from scipy.special import boxcox1p, inv_boxcox1p
import random
import lightgbm as lgb


def extract_max_min_for_every(each_name):
    t1 = pd.read_csv(os.path.join(groundtruth_dir, each_name))
    max_min = t1.describe().T
    max_min = max_min[['max', 'min']].reset_index(drop=False)
    max_min.rename(columns={'index': 'column_name', 'max': 'var_max', 'min': 'var_min'}, inplace=True)
    max_min['name'] = each_name
    temp_naidx = naidx[naidx['name'] == each_name]
    t1.reset_index(drop=False, inplace=True)
    t2 = pd.merge(temp_naidx, t1, on=['index'], left_index=False, right_index=False)
    t2['var_value'] = t2.apply(lambda x: x[x.column_name], axis=1)
    t2 = t2[['name', 'column_name', 'index', 'var_value']]
    return pd.merge(t2, max_min, left_index=False, right_index=False)


def init():
    global groundtruth_dir, missing_dir, real_test_dir, test_data_pred, \
        groudtruth_files, miss_files, use_columns, \
        ori_names, fun_list, column_names, naidx_testinfo, naidx
    # groundtruth路径
    groundtruth_dir = '/data/03.DACMI/data/train_groundtruth'
    # missing路径
    missing_dir = '/data/03.DACMI/data/train_with_missing'
    # 真实测试集路径
    real_test_dir = '/data/MissingData_Competition/test_data'
    # 填充结果保存路径
    test_data_pred = 'test_data_pred'
    #naidx.csv路径
    naidx_dir='/data/03.DACMI/naidx.csv'


    groudtruth_files = os.listdir(groundtruth_dir)
    miss_files = os.listdir(missing_dir)
    use_columns = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT',
                   'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

    def mylen(temp_list):
        temp_list = [temp_value for temp_value in temp_list if not np.isnan(temp_value)]
        return len(temp_list)

    fun_list = [max, min, np.mean, np.std, 'skew', mylen]
    fun_str = ['max', 'min', 'mean', 'std', 'skew', 'len']
    ##这里注意顺序一定不能乱，可以先读入groudtruth_label看看
    ori_names = ['CHARTTIME', 'PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV',
                 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']
    column_names = []
    for each_columns in ori_names:
        for each_fun in fun_str:
            column_names.append(each_columns + '_' + str(each_fun))

    # 重新生成特征，直接用diff会导致整个人该列diff都是NAN，不如使用真实值，而不用差
    for each_columns in ori_names:
        column_names.append(each_columns + '_last')
    # 加入反向diff
    for each_columns in ori_names:
        column_names.append(each_columns + '_next')
    # 加入二阶diff
    for each_columns in ori_names:
        column_names.append(each_columns + '_last2')
    # 加入二阶反向diff
    for each_columns in ori_names:
        column_names.append(each_columns + '_next2')
    # 加入三阶diff
    for each_columns in ori_names:
        column_names.append(each_columns + '_last3')
    # 加入三阶反向diff
    for each_columns in ori_names:
        column_names.append(each_columns + '_next3')

    naidx_testinfo_name = '/data/03.DACMI/naidx_testinfo.csv'
    if os.path.exists(naidx_testinfo_name):
        naidx_testinfo = pd.read_csv(naidx_testinfo_name)
    else:
        naidx = pd.read_csv(naidx_dir)
        naidx['pt.num'] = naidx['pt.num'].map(lambda x: str(x) + '.csv')
        naidx['i'] = naidx['i'].map(lambda x: x - 1)
        naidx.rename(columns={'pt.num': 'name', 'test': 'column_name', 'i': 'index'}, inplace=True)
        naidx_testinfo = pd.concat(
            Parallel(n_jobs=-1)(delayed(extract_max_min_for_every)(each_name) for each_name in groudtruth_files))
        naidx_testinfo.reset_index(drop=True, inplace=True)
        naidx_testinfo.to_csv(naidx_testinfo_name, index=None)
    print('init method finished')


def extract_time(each_df):
    each_df['time_24'] = each_df['CHARTTIME'] / 1440 % 24
    key = np.floor(each_df['CHARTTIME'] / 1440)
    dict_value = key.value_counts().sort_index()
    each_df['freq_count'] = [dict_value[np.floor(i)] for i in key.values]
    return each_df


def extract_trend(each_df):
    for iter2, each_column in enumerate(use_columns):
        each_column_index = each_df.columns.tolist().index(each_column)
        temp_list = []
        # 前面4个特殊处理，零用1，
        temp_list.append(1)
        temp_list.append(1)
        temp_list.append(each_df.iloc[1, each_column_index] / np.nanmean(each_df.iloc[0, each_column_index]))
        temp_list.append(each_df.iloc[2, each_column_index] / np.nanmean(each_df.iloc[0:2, each_column_index]))
        for i in range(4, each_df.shape[0]):
            temp_list.append(
                each_df.iloc[i - 1, each_column_index] / np.nanmean(each_df.iloc[i - 4:i - 1, each_column_index]))
        each_df[each_column + '_trend'] = temp_list
    return each_df


def read_data_and_extract_features(input_dir, each_name):
    if not each_name.endswith('.csv'):
        return pd.DataFrame()
    each_df = pd.read_csv(os.path.join(input_dir, each_name), engine='python')
    each_df['name'] = each_name
    ##生成index
    each_df.reset_index(drop=False, inplace=True)
    each_df['index_ratio'] = each_df['index'] / each_df.shape[0]
    each_tongji = pd.DataFrame()

    each_use = each_df.drop(['name', 'index', 'index_ratio'], axis=1)
    for i in range(each_df.shape[0]):
        if i == 0:
            each_tongji = each_tongji.append([each_use.iloc[:i + 1, :].agg(fun_list).values.T.flatten()])
        else:
            each_tongji = each_tongji.append([each_use.iloc[:i, :].agg(fun_list).values.T.flatten()])

    each_tongji = pd.concat([each_tongji.reset_index(drop=True), each_use.shift(1).reset_index(drop=True),
                             each_use.shift(-1).reset_index(drop=True),
                             each_use.shift(2).reset_index(drop=True),
                             each_use.shift(-2).reset_index(drop=True),
                             each_use.shift(3).reset_index(drop=True),
                             each_use.shift(-3).reset_index(drop=True), ], axis=1)
    each_tongji.columns = column_names

    each_df = extract_trend(extract_time(each_df.copy()))
    return pd.concat([each_df, each_tongji], axis=1)


def fillna(each_group):
    each_group.fillna(method='bfill', inplace=True)
    each_group.fillna(method='ffill', inplace=True)
    return each_group


def split_my_val2(group):
    mylen = len(group.index)
    return group.index[random.randint(0, mylen - 1)]


## 把评分方法加入验证过程
def cal_val(y_pre, d_train):
    global X_train2, X_val2, select_col
    ##不管是训练集还是验证集都是用训练集每个人的最大值和最小值
    y_label = d_train.get_label().values

    if select_col in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        y_pre = inv_boxcox1p(y_pre, 0)
        y_label = inv_boxcox1p(y_label, 0)
    elif select_col in ['PCRE']:
        y_pre = inv_boxcox1p(y_pre, -1.5)
        y_label = inv_boxcox1p(y_label, -1.5)
    if len(y_pre) == X_val2.shape[0]:
        mse = np.sum(np.square((y_label - y_pre) / (
                X_val2['max'] - X_val2['min'])))
        score = np.sqrt(mse / len(y_pre))
    elif len(y_pre) == X_train2.shape[0]:
        mse = np.sum(np.square((y_label - y_pre) / (
                X_train2['max'] - X_train2['min'])))
        score = np.sqrt(mse / len(y_pre))
    return 't1', score, False


def train_val_test(each_df, each_column, save_dir):
    global X_train2, X_val2, select_col
    select_col = each_column

    #     use_columns_temp=use_columns.copy()
    #     use_columns_temp.remove(each_column)
    #     combined_df=pd.DataFrame()
    #     for each_i,i in enumerate(use_columns_temp):
    #         combined_df[i+'**2']=each_df[i]**2
    #         combined_df[i+'**0.5']=each_df[i]**0.5
    #         #加减乘都不用做两次，除可以
    #         for each_j in range(each_i+1,len(use_columns)):
    #             j=use_columns[each_j]
    #             combined_df[i+'+'+j]=each_df[i]+each_df[j]
    #             combined_df[i+'-'+j]=each_df[i]-each_df[j]
    #             combined_df[i+'*'+j]=each_df[i]*each_df[j]
    #     for i in use_columns:
    #         #加减乘都不用做两次，除可以
    #         for j in use_columns:
    #             if i!=j:
    #                 combined_df[i+'/'+j]=each_df[i]/each_df[j]

    #     each_df=pd.concat([each_df,combined_df],axis=1)

    each_df.reset_index(drop=True, inplace=True)
    each_df.sort_values(['name', 'index'], inplace=True)
    each_df['chart_cos'] = np.cos(each_df['CHARTTIME'] / 1440)
    each_df['chart_sin'] = np.sin(each_df['CHARTTIME'] / 1440)
    each_df['timediff'] = each_df['CHARTTIME'] - each_df['CHARTTIME_last']

    if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        each_df[each_column] = boxcox1p(each_df[each_column], 0)
    elif each_column in ['PCRE']:
        each_df[each_column] = boxcox1p(each_df[each_column], -1.5)

    each_df_copy = each_df.drop([each_column], axis=1)

    each_max_min = naidx_testinfo[naidx_testinfo['column_name'] == each_column]

    # fillna
    # 在fillna前删除异常值
    if not each_column in ['HCT','HGB']:
        each_df_copy = pd.concat(
            Parallel(n_jobs=-1)(delayed(fillna)(each_group) for name, each_group in each_df_copy.groupby('name')))

    each_df_copy.set_index(['name', 'index'], drop=True, inplace=True)

    each_max_min.set_index(['name', 'index'], drop=True, inplace=True)
    each_df.set_index(['name', 'index'], drop=True, inplace=True)

    each_test = pd.merge(each_df_copy, each_max_min, left_index=True, right_index=True)

    each_train_x = each_df_copy.loc[each_df_copy.index.drop(each_test.index)]

    each_train_y = each_df.loc[each_train_x.index, each_column]

    each_test_x = each_test[each_train_x.columns]

    each_train_y = each_train_y.dropna()

    each_train_x = each_train_x.loc[each_train_y.index]

    X_train, X_val, y_train, y_val = train_test_split(each_train_x, each_train_y, test_size=0.3, random_state=2019)

    #     val_index=Parallel(n_jobs=1)(delayed(split_my_val2)(group)for name,group in each_train_x.groupby('name'))
    #     X_val=each_train_x.loc[val_index]
    #     y_val=each_train_y.loc[val_index]
    #     X_train=each_train_x.drop(val_index)
    #     y_train=each_train_y.loc[X_train.index]

    train_max_min = each_train_y.groupby('name').agg([max, min])
    # 需要恢复
    if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        train_max_min['max'] = inv_boxcox1p(train_max_min['max'], 0)
        train_max_min['min'] = inv_boxcox1p(train_max_min['min'], 0)
    elif each_column in ['PCRE']:
        train_max_min['max'] = inv_boxcox1p(train_max_min['max'], -1.5)
        train_max_min['min'] = inv_boxcox1p(train_max_min['min'], -1.5)
    X_train2 = pd.merge(X_train, train_max_min, left_index=True, right_index=True)
    X_val2 = pd.merge(X_val, train_max_min, left_index=True, right_index=True)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval = lgb.Dataset(X_val, y_val)  # 创建验证数据

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['mse'],
        'colsample_bytree': 0.9,
        'subsample': 0.9,

        'num_leaves': 30,  # 叶子节点个数
        'min_data': 50,  # 每个叶子节点最少样本数
        'max_depth': -1,  # 树深度
        'lambda_l2': 0.001,  # l2正则
        'lambda_l1': 0.01,  # l1正则
        'num_threads': 12,

        'verbose': -1, 'tree_learner': 'voting', 'seed': 2019
    }

    # 训练 cv and train
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_eval], feval=cal_val,
                    num_boost_round=3000,
                    early_stopping_rounds=100,
                    verbose_eval=100)

    each_test['pred'] = gbm.predict(each_test[each_train_x.columns])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    joblib.dump(gbm, os.path.join(save_dir, each_column))
    # 这里需要恢复原来的值计算score

    if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        each_test['pred'] = inv_boxcox1p(each_test['pred'], 0)
    elif each_column in ['PCRE']:
        each_test['pred'] = inv_boxcox1p(each_test['pred'], -1.5)

    mse = np.sum(np.square((each_test['var_value'] - each_test['pred']) / (
            each_test['var_max'] - each_test['var_min'])))
    score = np.sqrt(mse / each_test.shape[0])
    return score


# 单个人预测，很慢
def test_for_predict(each_dir, each_name):
    # 生成特征
    each_df = read_data_and_extract_features(each_dir, each_name)
    to_each_df = pd.DataFrame()
    each_df['chart_cos'] = np.cos(each_df['CHARTTIME'] / 1440)
    each_df['chart_sin'] = np.sin(each_df['CHARTTIME'] / 1440)
    each_df['timediff'] = each_df['CHARTTIME'] - each_df['CHARTTIME_last']
    each_df.set_index(['name', 'index'], drop=True, inplace=True)
    each_df_fillna = pd.concat(
        Parallel(n_jobs=-1)(delayed(fillna)(each_group) for name, each_group in each_df.groupby('name')))

    for each_column in use_columns:
        # 保留预测位置
        y_index = each_df[each_column].isna()
        each_test_x = each_df.drop(each_column, axis=1)

        # if each_column in ['HCT', 'HGB']:
        #     each_model = joblib.load(os.path.join('lgb统计', each_column))
        #     each_test_x = each_df.drop(each_column, axis=1)
        # else:
        each_model = joblib.load(os.path.join('lgb统计加fillna', each_column))
        each_test_x = each_df_fillna.drop(each_column, axis=1)

        each_test_x = each_test_x[y_index]
        each_test_not_x = each_df.drop(each_test_x.index)

        each_test_x[each_column] = each_model.predict(each_test_x)
        # 这里需要恢复原来的值计算score

        if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
            each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], 0)
        elif each_column in ['PCRE']:
            each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], -1.5)
        each_test_x = pd.concat([each_test_not_x, each_test_x])
        to_each_df[each_column] = each_test_x[each_column]
    if not os.path.exists('fillna_test_data'):
        os.mkdir('fillna_test_data')
    to_each_df['CHARTTIME'] = each_df['CHARTTIME']
    to_each_df.sort_values(['index'], inplace=True)
    to_each_df.to_csv(os.path.join('fillna_test_data', each_name), index=None)


def predict_for_column(each_df, each_df_fillna, each_column):
    # 保留测试集index
    y_index = each_df[each_column].isna()

    # if each_column in ['HCT', 'HGB']:
    #     each_model = joblib.load(os.path.join('lgb统计', each_column))
    #     each_test_x = each_df.drop(each_column, axis=1)
    # else:
    each_model = joblib.load(os.path.join('lgb统计加fillna', each_column))
    each_test_x = each_df_fillna.drop(each_column, axis=1)

    each_test_x = each_test_x[y_index]
    each_test_not_x = each_df.drop(each_test_x.index)

    each_test_x[each_column] = each_model.predict(each_test_x)
    # 这里需要恢复原来的值计算score

    if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], 0)
    elif each_column in ['PCRE']:
        each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], -1.5)
    each_test_x = pd.concat([each_test_not_x, each_test_x])
    return each_test_x[each_column]


# 所有人预测，很快
def test_for_predict_once(each_df):
    each_df['chart_cos'] = np.cos(each_df['CHARTTIME'] / 1440)
    each_df['chart_sin'] = np.sin(each_df['CHARTTIME'] / 1440)
    each_df['timediff'] = each_df['CHARTTIME'] - each_df['CHARTTIME_last']
    each_df.set_index(['name', 'index'], drop=True, inplace=True)

    each_df_fillna = pd.concat(
        Parallel(n_jobs=-1)(delayed(fillna)(each_group) for name, each_group in each_df.groupby('name')))
    ##一次做完fillna,然后按照索引对应
    to_each_df = pd.concat(Parallel(n_jobs=len(use_columns))(
        delayed(predict_for_column)(each_df.copy(), each_df_fillna.copy(), each_col) for each_col in use_columns),
        axis=1)
    to_each_df['CHARTTIME'] = each_df['CHARTTIME']
    to_each_df.sort_values(['name', 'index'], inplace=True)
    if not os.path.exists(test_data_pred):
        os.mkdir(test_data_pred)
    _ = to_each_df.groupby('name').apply(lambda x: x.to_csv(os.path.join(test_data_pred, x.name), index=None))
    # return to_each_df


def compute_score_for_mytest(each_df):
    each_df['chart_cos'] = np.cos(each_df['CHARTTIME'] / 1440)
    each_df['chart_sin'] = np.sin(each_df['CHARTTIME'] / 1440)
    each_df['timediff'] = each_df['CHARTTIME'] - each_df['CHARTTIME_last']
    each_df.set_index(['name', 'index'], drop=True, inplace=True)

    each_df_fillna = pd.concat(
        Parallel(n_jobs=-1)(delayed(fillna)(each_group) for name, each_group in each_df.groupby('name')))

    for each_column in use_columns:
        # load模型
        each_test_x = each_df[each_df[each_column].notna()]
        each_max_min = each_df.groupby('name')[each_column].agg([max, min])

        if each_column in ['HCT', 'HGB']:
            each_model = joblib.load(os.path.join('lgb统计', each_column))
            each_test_x = each_df.drop(each_column, axis=1)
        else:
            each_model = joblib.load(os.path.join('lgb统计加fillna', each_column))
            each_test_x = each_df_fillna.drop(each_column, axis=1)

        ###生成mytest
        mytest_index = Parallel(n_jobs=1)(delayed(split_my_val2)(group) for name, group in each_test_x.groupby('name'))
        each_test_x = each_test_x.loc[mytest_index]
        each_test_y = each_df.loc[mytest_index][each_column]

        each_test_x[each_column] = each_model.predict(each_test_x)
        # 这里需要恢复原来的值计算score

        if each_column in ['PBUN', 'PGLU', 'WBC', 'PLT']:
            each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], 0)
        elif each_column in ['PCRE']:
            each_test_x[each_column] = inv_boxcox1p(each_test_x[each_column], -1.5)

        mse = np.sum(np.square((each_test_y - each_test_x[each_column]) / (
                each_max_min['max'] - each_max_min['min'])))
        score = np.sqrt(mse / each_test_x.shape[0])
        print(each_column, score)
        return score


def cal_loss(y_pre, d_train):
    global X_train2, X_val2, select_col
    ##需要判断此时是训练集还是验证集来决定取var_max和var_min
    y_label = d_train.get_label()
    if select_col in ['PBUN', 'PGLU', 'WBC', 'PLT']:
        y_pre = inv_boxcox1p(y_pre, 0)
        y_label = inv_boxcox1p(y_label, 0)
    elif select_col in ['PCRE']:
        y_pre = inv_boxcox1p(y_pre, -1.5)
        y_label = inv_boxcox1p(y_label, -1.5)

    if len(y_pre) == X_train2.shape[0]:
        mse = (y_pre - y_label) / (
                X_train2['var_max'] - X_train2['var_min'])
    elif len(y_pre) == X_val2.shape[0]:
        mse = (y_pre - y_label) / (
                X_val2['var_max'] - X_val2['var_min'])
    #     grad = y_pre - y_label
    hess = np.power(np.abs(mse), 0.5)
    return mse, hess


def main():
    init()
    missing_data_name = 'ori_missing_tongji_train_time_trend.data'
    test_data_name = 'real_test_with_tongji.data'

    if os.path.exists(missing_data_name):
        ori_missing_tongji_train_time_trend = joblib.load(missing_data_name)
    else:
        ori_missing_tongji_train_time_trend = pd.concat(Parallel(n_jobs=-1)(
            delayed(read_data_and_extract_features)(missing_dir, each_name) for each_name in miss_files))
        joblib.dump(ori_missing_tongji_train_time_trend, missing_data_name)

    # 训练并保存模型,顺序训练可以观察详细loss
    if len(os.listdir('lgb统计加fillna'))!=13:
        trend_score = {}
        for each_column in use_columns:
            trend_score[each_column] = train_val_test(ori_missing_tongji_train_time_trend.copy(), each_column,
                                                      'lgb统计加fillna')
    print('train finished')
    del ori_missing_tongji_train_time_trend
    # 读入测试集
    if os.path.exists(test_data_name):
        real_test_with_tongji = joblib.load(test_data_name)
    else:
        real_test_files = os.listdir(real_test_dir)
        real_test_with_tongji = Parallel(n_jobs=-1)(
            delayed(read_data_and_extract_features)(real_test_dir, each_name) for each_name in real_test_files)
        real_test_with_tongji = pd.concat(real_test_with_tongji)
        joblib.dump(real_test_with_tongji, test_data_name)

    test_for_predict_once(real_test_with_tongji)


if __name__ == '__main__':
    main()
