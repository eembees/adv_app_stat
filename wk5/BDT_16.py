import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from lib.lib_math import bin_data
import sklearn.model_selection as ms
from pathlib import Path

df_big = pd.read_csv('BDT_16var.txt',
                     header=None, dtype=float, delim_whitespace=True,
                     # usecols=['l{}'.format(i) for i in range(16)]
                     )
df_big.rename('l{}'.format, axis='columns',inplace=True)
# print(df_big.head())
# print(df_big.keys())
# exit()

df_big['label'] = [0, 1] * 5000

df_train, df_test = ms.train_test_split(df_big, test_size=0.2)

df_train_label = df_train['label']
df_test_label = df_test['label']

dtrain = xgb.DMatrix(df_train.drop(columns='label'), label=df_train['label'])
dtest = xgb.DMatrix(df_test.drop(columns='label'), label=df_test['label'])

num_round = 10

param = {
    # 'max_depth': 10,
    'eta': 1,
    'verbosity': 0,
    # 'objective': 'binary:logitraw',
    'objective': 'binary:logistic',
    'tree_method':'exact',
}

bst = xgb.train(param, dtrain, num_boost_round=num_round, )
# bst = xgb.train(param, dtrain, num_round, evallist)

bst.dump_model('dump.raw.txt')

yprobs = bst.predict(dtest)

y_pred = yprobs > 0.5

# print(y_pred)

y_val = np.array(np.equal(y_pred, df_test_label.values), dtype=int)


# print(np.sum(y_val))
# print(len(y_val))

print(np.sum(y_val) / len(y_val))



# exit()
#
# fig2, ax2 = plt.subplots(figsize=(6,10))
# xgb.plot_tree(bst, ax=ax2, rankdir='LR')
# fig2.tight_layout()
# fig2.savefig(Path('/Users/mag/Desktop/Figure_1.pdf'), dpi=1000)
# print(Path('/Users/mag/Desktop/Figure_1.pdf').exists())
#
# fig, ax = plt.subplots()
#
# bin_centers, counts, unc, mask = bin_data(yprobs, N_bins=25)
#
# ax.errorbar(bin_centers[mask], counts[mask], yerr=unc[mask], )
#
# fig.tight_layout()
# plt.show()
