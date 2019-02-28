import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from lib.lib_math import bin_data

# read data
df_bg_train = pd.read_csv('../wk5/BDT_background_train.txt', delim_whitespace=True, dtype=float, names=['a', 'b', 'c'],
                          header=None)
df_sg_train = pd.read_csv('../wk5/BDT_signal_train.txt', delim_whitespace=True, dtype=float, names=['a', 'b', 'c'],
                          header=None)

df_bg_test = pd.read_csv('../wk5/BDT_background_test.txt', delim_whitespace=True, dtype=float, names=['a', 'b', 'c'],
                         header=None)
df_sg_test = pd.read_csv('../wk5/BDT_signal_test.txt', delim_whitespace=True, dtype=float, names=['a', 'b', 'c'],
                         header=None)

# assign labels
df_bg_train['label'] = 0
df_bg_test['label'] = 0
df_sg_train['label'] = 1
df_sg_test['label'] = 1

df_train = pd.concat([df_bg_train, df_sg_train])
df_test = pd.concat([df_bg_test, df_sg_test])

df_train_label = df_train['label']
df_test_label = df_test['label']

dtrain = xgb.DMatrix(df_train.drop(columns='label'), label=df_train['label'])
dtest = xgb.DMatrix(df_test.drop(columns='label'), label=df_test['label'])

num_round = 10

param = {
    # 'max_depth': 2,
    # 'eta': 1,
    'verbosity': 0,
    # 'objective': 'binary:logitraw'
    'objective': 'binary:logistic'
}

bst = xgb.train(param, dtrain, num_boost_round=num_round, )
# bst = xgb.train(param, dtrain, num_round, evallist)

bst.dump_model('dump.raw.txt')

yprobs = bst.predict(dtest)

y_pred = yprobs > 0.5

# print(y_pred)

y_val = np.array(np.equal(y_pred, df_test_label.values), dtype=int)
print(np.sum(y_val))
print(len(y_val))

print(y_val)

fig, ax = plt.subplots()

bin_centers, counts, unc, mask = bin_data(yprobs, N_bins=25)

ax.errorbar(bin_centers[mask], counts[mask], yerr=unc[mask], )

fig.tight_layout()
plt.show()
