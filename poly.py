import pathlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_DIR = pathlib.Path("data/")

train_data = pd.read_csv(DATA_DIR.joinpath("activity_train.csv.zip"),
                         parse_dates=["date"], index_col="date",
                         compression="zip")
test_data = pd.read_csv(DATA_DIR.joinpath("activity_test.csv.zip"),
                        parse_dates=["date"], index_col="date",
                        compression="zip")
activity_test_target = pd.read_csv(DATA_DIR.joinpath("activity_test_timestamps.csv"),
                                   index_col="date",
                                   parse_dates=["date"])


data = pd.concat([train_data[test_data.columns], test_data])
train_targets = train_data[["activity", "atactic_1", "atactic_2", "atactic_3"]].copy()


dd = [30, 60, 90, 120, 180, 240, 360, 720]
st = ["mean", "median", "std", "max", "min"]

rows = []
i = 0
for dt in list(train_data.index) + list(test_data.index):
    i += 1
    if i < 300000:
        continue
    try:
        target = train_data[dt:dt+timedelta(minutes=6*60)]['activity'].values[-1]
    except:
        target = 0
    features = []
    fnames = []
    for d in dd:
        df_gapped = data[(dt - timedelta(minutes=d)):dt]
        for m in st:
            foo = df_gapped.apply(m, axis=0)
            features += list(foo.values)
            fnames += ['{}_{}_{}'.format(e, d, m) for e in foo.index]
    rows.append([dt] + features + [target])
    if i % 180 == 31:
        print(dt)

df = pd.DataFrame([e[1:] for e in rows], columns=fnames + ['y'], dtype=np.float32).fillna(0)        
df['dt'] = [e[0] for e in rows]

df['m'] = df['dt'].apply(lambda x: int(str(x)[-5:-3]) // 12)
df['h'] = df['dt'].apply(lambda x: int(str(x)[-8:-6]))
extra_f = ['m', 'h']
# for i in range(0, 24):
#     df['h' + str(i)] = (df['h'] == i).astype(int)
#     extra_f.append('h' + str(i))
# for i in range(0, 12):
#     df['m' + str(i)] = (df['m'] == i).astype(int)
#     extra_f.append('m' + str(i))

print(df.info())

top_f = ["f55_60_std", "f35_720_max", "f53_60_std", "f18_240_mean", "f53_30_std", "f0_720_std", "f54_60_std", "f36_720_min", "f18_360_mean", "f54_30_std", "f14_360_mean", "f7_30_min", "f54_120_max", "f14_360_min", "f9_720_mean", "f50_360_max", "f7_60_min", "f6_360_max", "f14_240_min", "f45_720_median", "f32_720_median", "f30_720_max", "f20_240_min", "f41_720_median", "f54_60_max", "f53_360_max", "f4_720_max", "f13_360_max", "f6_720_max", "f20_720_max", "f7_720_median", "f7_240_mean", "f15_720_max", "f0_360_std", "f9_720_std", "f29_180_std", "f24_180_std", "f22_360_max", "f4_180_min", "f21_720_mean", "f0_240_std", "f29_360_min", "f4_360_median", "f55_720_min", "f53_720_max", "f44_720_median", "f55_90_std", "f0_720_median", "f0_240_max", "f18_720_mean", "f31_720_max", "f54_90_max", "f27_360_max", "f37_120_min", "f4_30_min", "f44_720_std", "f15_720_std", "f50_720_std", "f30_240_median", "f44_240_std", "f35_180_mean", "f39_360_median", "f36_120_max", "f10_360_max", "f17_360_median", "f46_30_min", "f5_240_min", "f13_720_median", "f7_30_max", "f6_720_median", "f7_360_min", "f7_30_median", "f37_720_std", "f3_720_median", "f36_720_std", "f7_180_min", "f6_180_max", "f17_720_median", "f4_720_median", "f46_720_mean", "f49_240_max", "f54_90_std", "f4_720_mean", "f20_120_min", "f17_720_std", "f18_240_median", "f16_240_min", "f45_360_max", "f55_120_max", "f20_360_min", "f25_720_median", "f4_30_median", "f4_30_mean", "f12_30_mean", "f4_60_median", "f4_60_min", "f24_120_std", "f4_120_min", "f4_180_max", "f42_240_std", "f45_120_median", "f30_180_median", "f30_360_mean", "f15_240_max", "f46_30_max", "f24_240_std", "f9_720_min", "f14_30_min", "f12_360_median", "f30_120_min", "f33_720_std", "f8_360_min", "f1_240_std", "f4_240_min", "f14_60_max", "f0_360_mean", "f15_360_min", "f5_180_min", "f3_360_mean", "f26_720_max", "f33_720_max", "f23_360_max", "f44_360_min", "f18_720_min", "f29_30_max", "f42_720_mean", "f16_720_min", "f53_60_min", "f29_30_mean", "f13_720_min", "f29_720_mean", "f45_720_mean", "f6_240_max", "f38_360_max", "f3_720_max", "f54_90_mean", "f55_240_max", "f31_240_max", "f35_90_median", "f49_360_max", "f36_360_std", "f53_360_min", "f14_360_std", "f12_180_median", "f6_360_min", "f53_90_std", "f54_30_mean", "f40_360_max", "f36_720_mean", "f27_720_max", "f54_30_min", "f21_360_min", "f25_720_min", "f41_720_max", "f0_180_max", "f31_360_max", "f43_720_min", "f19_360_max", "f53_180_min", "f35_120_median", "f41_360_median", "f55_30_std", "f44_720_min", "f50_240_min", "f12_120_std", "f5_360_min", "f5_240_std", "f35_240_min", "f33_240_median", "f4_90_median", "f23_180_mean", "f8_180_mean", "f6_180_median", "f16_720_std", "f12_90_median", "f15_120_mean", "f15_180_std", "f19_60_mean", "f16_360_std", "f51_90_max", "f15_360_std", "f34_90_min", "f9_30_min", "f6_180_std", "f42_180_std", "f4_90_min", "f19_120_mean", "f29_360_mean", "f14_30_max", "f34_720_max", "f34_120_min", "f45_90_max", "f54_240_std", "f44_120_std", "f54_360_max", "f24_360_std", "f13_180_min", "f12_120_max", "f33_180_max", "f12_720_std", "f6_180_mean", "f27_240_median", "f15_60_mean", "f44_180_std", "f12_120_median", "f27_720_mean", "f52_240_min", "f17_360_min", "f12_60_median", "f4_360_max", "f44_720_mean", "f20_180_max", "f13_360_median", "f5_720_min", "f14_30_median", "f20_240_std", "f30_180_min", "f29_720_median", "f5_240_mean", "f55_360_min", "f42_720_min", "f35_720_min", "f36_240_median", "f31_180_min", "f55_720_std", "f23_720_mean", "f22_720_median", "f31_60_max", "f2_360_min", "f55_90_max", "f12_360_min", "f51_30_median", "f52_30_min", "f15_240_std", "f49_720_max", "f45_720_max", "f55_90_mean", "f11_240_max", "f47_720_mean", "f6_360_median", "f21_240_max", "f54_120_std", "f45_90_std", "f46_360_mean", "f14_720_mean", "f20_60_min", "f50_240_max", "f31_720_median", "f3_60_median", "f51_360_mean", "f48_360_max", "f3_360_max", "f46_30_median", "f52_360_min", "f4_30_max", "f47_360_max", "f40_240_std", "f10_720_mean", "f29_30_min", "f0_180_std", "f27_720_std", "f53_90_max", "f0_60_std", "f2_720_median", "f34_180_max", "f13_120_std", "f1_720_mean", "f8_360_std", "f36_360_min", "f37_720_median", "f47_240_max", "f49_720_min", "f8_120_std", "f30_120_median", "f40_720_median", "f53_60_max", "f0_360_max", "f7_30_mean", "f9_720_median", "f42_120_min", "f18_720_std", "f35_60_max", "f54_180_std", "f25_240_median", "f5_720_std", "f32_360_min", "f10_720_median", "f14_180_min", "f41_360_std", "f53_90_mean", "f21_240_mean", "f5_720_median", "f21_120_min", "f26_180_std", "f49_720_median", "f16_90_mean", "f48_720_median", "f1_360_min", "f0_30_min"]

df = df[top_f + ['y', 'dt']]
print(df.info())

df.to_csv('df.tsv', index=False, sep='\t')

df_train = df[(df['dt'] > '2018') & (df['dt'] < '2018-12') & (df['y'] != 0)]#.copy()
df_train_all = df[(df['dt'] > '2018') & (df['y'] != 0)]#.copy()
df_test = df[(df['dt'] >= '2018-12') & (df['y'] != 0)]#.copy()# & (df['dt'] < '2017-10')].copy()
df_pred = df[(df['dt'] >= '2019')]#.copy()# & (df['dt'] < '2017-10')].copy()
print(len(df_train), len(df_train_all), len(df_test), len(df_pred))



def xgb_mape(preds, dtrain):
    labels = dtrain.get_label()
    return('mape', np.mean(np.abs((labels - preds) / labels)))


def learn_xgb_models():
    xgb_params = {
        'objective': 'reg:linear',
        'max_depth': 4, 
        'min_child_weight': 10, 
        'learning_rate': 0.05,
        'colsample_bylevel':  0.2,
        'silent': False,
        'seed': 0,
        'booster': 'gbtree',
        'alpha': 2000,
        'beta': 1,
    }

    xgb_params_linear = {
        'objective': 'reg:linear',
        'max_depth': 3, 
        'min_child_weight': 10, 
        'learning_rate': 0.1,
        'colsample_bylevel':  0.3,
        'silent': False,
        'seed': 0,
        'booster': 'gblinear',
        'alpha': 20000,
        'beta': 5,
        'nthread': 1,
    }

    models_all = []
    models_all_l = []
    models = []
    models_l = []

    fnames = [e for e in top_f if e not in extra_f]# + extra_f
    #fnames = fnames_0.copy() + extra_f
    fnames_l = fnames.copy()# + extra_f


    for i in range(7):
        xgb_params['seed'] = i
        xgb_params_linear['seed'] = i

        num_rounds = 150
        num_rounds_l = 80

        dtrain = xgb.DMatrix(df_train[fnames], df_train['y'])
        dtrain_all = xgb.DMatrix(df_train_all[fnames], df_train_all['y'])
        dtest = xgb.DMatrix(df_test[fnames], df_test['y'])
        dpred = xgb.DMatrix(df_pred[fnames])

        dtrain_l = xgb.DMatrix(df_train[fnames_l], df_train['y'])
        dtrain_all_l = xgb.DMatrix(df_train_all[fnames_l], df_train_all['y'])
        dtest_l = xgb.DMatrix(df_test[fnames_l], df_test['y'], )
        dpred_l = xgb.DMatrix(df_pred[fnames_l])

        watchlist  = [(dtest, 'test'), (dtrain, 'train')]
        watchlist_l  = [(dtest_l, 'test'), (dtrain_l, 'train')]

        evals_result = {}
        xgb_model = xgb.train(xgb_params, 
                              dtrain, num_rounds, watchlist, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)
#         xgb_model_linear = xgb.train(xgb_params_linear, 
#                                      dtrain_l, num_rounds_l, watchlist_l, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)

        models.append(xgb_model)
#        models_l.append(xgb_model_linear)

        xgb_model_all = xgb.train(xgb_params, 
                              dtrain_all, num_rounds, watchlist, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)
#         xgb_model_linear_all = xgb.train(xgb_params_linear, 
#                                      dtrain_all_l, num_rounds_l, watchlist_l, 
#                                          feval=xgb_mape, evals_result=evals_result, verbose_eval=25)

        models_all.append(xgb_model_all)
#         models_all_l.append(xgb_model_linear_all)

        print('=========' + str(i))
    
    return dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l


dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l = learn_xgb_models()


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))

def inference(dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l):
    ntree = 77#100
    alpha = 1.0
    tr_pred =  alpha * models[0].predict(dtrain, ntree_limit=ntree)# + (1 - alpha) * models_l[0].predict(dtrain_l)
    te_pred = alpha * models[0].predict(dtest, ntree_limit=ntree)# + (1 - alpha) * models_l[0].predict(dtest_l)
    for i in range(1, len(models)):
        tr_pred += alpha * models[i].predict(dtrain, ntree_limit=ntree)# + (1 - alpha) * models_l[i].predict(dtrain_l)
        te_pred += alpha * models[i].predict(dtest, ntree_limit=ntree)# + (1 - alpha) * models_l[i].predict(dtest_l)
    tr_pred /= len(models)
    te_pred /= len(models)

    pred_pred = alpha * models_all[0].predict(dpred, ntree_limit=ntree)# + (1 - alpha) * models_all_l[0].predict(dpred_l)
    for i in range(1, len(models)):
        pred_pred += alpha * models_all[i].predict(dpred, ntree_limit=ntree)# + (1 - alpha) * models_all_l[i].predict(dpred_l)
    pred_pred /= len(models_all)    

    df_res = df_pred[['dt']].copy()
    df_res['pred'] = pred_pred    
    df_res = df_res[df_res['dt'].isin(activity_test_target.index)]
    df_res.columns = ['date', 'activity']
    return df_res


res = inference(dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l)
res.to_csv('poly_submit.csv', index=False)
