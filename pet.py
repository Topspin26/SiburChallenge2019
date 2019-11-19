import pathlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_DIR = pathlib.Path("data/")

train_target = pd.read_csv(DATA_DIR.joinpath("pet_target_train.csv"), index_col="date", parse_dates=["date"])
test_target = pd.read_csv(DATA_DIR.joinpath("pet_test_timestamps.csv"), index_col="date", parse_dates=["date"])
daily = pd.read_csv(DATA_DIR.joinpath("pet_daily.csv"), index_col="date", parse_dates=["date"])
weekly = pd.read_csv(DATA_DIR.joinpath("pet_weekly.csv"), index_col="date", parse_dates=["date"])


daily['dt'] = daily.index
wfts = weekly.resample("D").ffill()
wfts['dt'] = wfts.index
fts = daily.merge(wfts, on=['dt'])

tt = train_target.resample("D").ffill()
tt['dt'] = tt.index
fts = fts.merge(tt, how='left')

data = fts
data = data.set_index(data['dt'])



dd = [7, 14, 28, 35, 42, 49, 56, 90, 180]
st = ["mean", "median", "std", "max", "min"]

rows = []
for dt in pd.date_range('2007-01-01', '2019-07-01'):
    if str(dt)[8:10] != '01':
        continue
    target = data[dt:(dt+timedelta(30))]['pet'].mean()
    features = []
    fnames = []
    for d in dd:
        df_gapped = data[(dt - timedelta(23 + d)):(dt - timedelta(23))]
        for m in st:
            foo = df_gapped.apply(m, axis=0)
            features += list(foo.values)
            fnames += ['{}_{}_{}'.format(e, d, m) for e in foo.index]
    rows.append(features + [target])
    print(dt, target)



df = pd.DataFrame(rows, columns=fnames + ['y']).fillna(0)
df['dt'] = [e for e in pd.date_range('2007-01-01', '2019-07-01') if str(e)[8:10].endswith('01')]

df['m'] = df['dt'].apply(lambda x: int(str(x)[5:7]))
extra_f = ['m']
for i in range(1, 13):
    df['m' + str(i)] = (df['m'] == i).astype(int)
    extra_f.append('m' + str(i))

fnames = [e for e in fnames if not e.startswith('dt') and not e.startswith('pet')]

df_train = df[(df['dt'] > '2000') & (df['dt'] < '2013')].copy()
df_train_all = df[(df['dt'] > '2000') & (df['y'] != 0)].copy()
df_test = df[(df['dt'] >= '2013') & (df['y'] != 0)].copy()# & (df['dt'] < '2017-10')].copy()
df_pred = df[(df['dt'] >= '2016')].copy()# & (df['dt'] < '2017-10')].copy()
print(len(df_train), len(df_train_all), len(df_test), len(df_pred))


def xgb_mape(preds, dtrain):
    labels = dtrain.get_label()
    return('mape', np.mean(np.abs((labels - preds) / labels)))


def learn_xgb_models():

    xgb_params = {
        'objective': 'reg:linear',
        'max_depth': 2, 
        'min_child_weight': 10, 
        'learning_rate': 0.05,
        'colsample_bylevel':  0.5,
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

    for i in range(20):
        xgb_params['seed'] = i
        xgb_params_linear['seed'] = i

        num_rounds = 40
        num_rounds_l = 80

        dtrain = xgb.DMatrix(df_train[fnames + extra_f], df_train['y'])
        dtrain_all = xgb.DMatrix(df_train_all[fnames + extra_f], df_train_all['y'])
        dtest = xgb.DMatrix(df_test[fnames + extra_f], df_test['y'])
        dpred = xgb.DMatrix(df_pred[fnames + extra_f])

        dtrain_l = xgb.DMatrix(df_train[fnames + extra_f], df_train['y'])
        dtrain_all_l = xgb.DMatrix(df_train_all[fnames + extra_f], df_train_all['y'])
        dtest_l = xgb.DMatrix(df_test[fnames + extra_f], df_test['y'])
        dpred_l = xgb.DMatrix(df_pred[fnames + extra_f])

        watchlist  = [(dtest, 'test'), (dtrain, 'train')]
        watchlist_l  = [(dtest_l, 'test'), (dtrain_l, 'train')]

        evals_result = {}
        xgb_model = xgb.train(xgb_params, 
                              dtrain, num_rounds, watchlist, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)
        xgb_model_linear = xgb.train(xgb_params_linear, 
                                     dtrain_l, num_rounds_l, watchlist_l, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)

        models.append(xgb_model)
        models_l.append(xgb_model_linear)

        xgb_model_all = xgb.train(xgb_params, 
                              dtrain_all, num_rounds, watchlist, feval=xgb_mape, evals_result=evals_result, verbose_eval=25)
        xgb_model_linear_all = xgb.train(xgb_params_linear, 
                                     dtrain_all_l, num_rounds_l, watchlist_l, 
                                         feval=xgb_mape, evals_result=evals_result, verbose_eval=25)

        models_all.append(xgb_model_all)
        models_all_l.append(xgb_model_linear_all)

        print('=========' + str(i))    
    
    return dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l


dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l = learn_xgb_models()


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))

def inference(dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l):
    tr_pred =  0.5 * models[0].predict(dtrain) + 0.5 * models_l[0].predict(dtrain_l)
    te_pred = 0.5 * models[0].predict(dtest) + 0.5 * models_l[0].predict(dtest_l)
    for i in range(1, len(models)):
        tr_pred += 0.5 * models[i].predict(dtrain) + 0.5 * models_l[i].predict(dtrain_l)
        te_pred += 0.5 * models[i].predict(dtest) + 0.5 * models_l[i].predict(dtest_l)
    tr_pred /= len(models)
    te_pred /= len(models)

    pred_pred = 0.5 * models_all[0].predict(dpred) + 0.5 * models_all_l[0].predict(dpred_l)
    for i in range(1, len(models)):
        pred_pred += 0.5 * models_all[i].predict(dpred) + 0.5 * models_all_l[i].predict(dpred_l)
    pred_pred /= len(models_all)

    df_test['month'] = df_test['dt'].apply(lambda x: str(x)[:8] + '01')
    df_test['pred'] = te_pred

    dff = df_test.drop_duplicates(['month'])[['month', 'dt', 'y', 'pred']]

    print(mape(dff['y'], dff['pred']))

    df_pred['pred'] = pred_pred
    df_pred['month'] = df_pred['dt'].apply(lambda x: str(x)[:8] + '01')
    dff = df_pred.drop_duplicates(['month'])[['month', 'dt', 'y', 'pred']]
    res = dff[['month', 'pred']].copy()
    res.columns = ['date', 'pet']
    
    return res


res = inference(dtrain, dtrain_all, dtest, dpred, dtrain_l, dtrain_all_l, dtest_l, dpred_l, models, models_l, models_all, models_all_l)
res.to_csv('pet_submit.csv', index=False)
