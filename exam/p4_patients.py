import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lib.lib_math as lm
import lib.nice_string_output as ns


# importing specific packages for this problem
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn import linear_model
from sklearn.svm import l1_min_c

import xgboost as xgb


# reading data

df_train = pd.read_csv('../exam/Exam_2019_Prob4_TrainData.csv')
df_test = pd.read_csv('../exam/Exam_2019_Prob4_TestData.csv')
df_blind = pd.read_csv('../exam/Exam_2019_Prob4_BlindData.csv')

# print(df_test.keys())

features_to_train = ['Gender', 'ScheduledDay',
'AppointmentDay', 'Age',
'TimeDifference',
'Neighbourhood', 'Diabetes', 'Alcoholism',
'Handcap', 'SMS_received', 'R1'
]
names = features_to_train.copy()

categorical_features = [
'Gender',
'ScheduledDay',
'AppointmentDay',
'Neighbourhood',
'Diabetes',
'Alcoholism',
'Handcap',
'SMS_received',
'R1'
]

numerical_features = [feature for feature in features_to_train if feature not in categorical_features]

# print(numerical_features)
# for feat in features_to_train:
#     print(feat)
#     print(df_train[feat].unique())
# make a column transformer
ct = ColumnTransformer(
    [('ohe', OneHotEncoder(categories='auto'), categorical_features),
     ('norm', Normalizer(norm='l1'), numerical_features)]
)

# prepare data
# TODO run on full dataset
X_train = ct.fit_transform(df_train[features_to_train])
# X_train = df_train[features_to_train].values
X_test  = df_train[features_to_train].values
X_blind  = df_train[features_to_train].values

# y_train = ct.fit_transform(df_train['No-show'])
# y_train = df_train['No-show'].values
y_test = df_test['No-show'].values

print(type(X_train))
exit()
# print(X_train[0].shape)

# # data visualization
# figh, axh = plt.subplots()
# axh = df_train.hist()#by=df_train['No-show'])
# # figh.tight_layout()
# plt.savefig('./figs/p4_feature_visualization.png')

# figv, axesv = plt.subplots(nrows=3, ncols=4, figsize=(10,6))
# axv = axesv.ravel()
#
# for i, feature in enumerate(features_to_train):
#     axv[i].hist(df_train[feature].values)
#     axv[i].set_title(feature)
#
# figv.savefig('./figs/p4_feature_visualization.png', dpi=300)
# g = sns.pairplot(df_train.drop(columns='ID'), hue='No-show', diag_kind='hist')
#
# g.savefig('./figs/p4_feature_visualization.png', dpi=300)
# exit()
# the neighborhood variable should be one out of k encoded
#
# # Define which columns should be encoded vs scaled
# columns_to_encode = ['Neighbourhood']
# columns_to_scale  = ['gre', 'gpa']
#
# # Instantiate encoder/scaler
# scaler = StandardScaler()
# ohe    = OneHotEncoder(sparse=False)
#
# # Scale and Encode Separate Columns
# scaled_columns  = scaler.fit_transform(dataset[columns_to_scale])
# encoded_columns =    ohe.fit_transform(dataset[columns_to_encode])
#
# # Concatenate (Column-Bind) Processed Columns Back Together
# processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
#

# classifier with lasso no CV
n_coefs = 10

cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 5, 16)

clf = linear_model.LogisticRegression(penalty='l1', solver='saga',
                                      tol=1e-6, max_iter=int(1e6),
                                      warm_start=True, fit_intercept=False)

coefs_ = []
scores = []
for ic, c in enumerate(cs):
    print('fitting with c {}'.format(ic))
    print(c)
    clf.set_params(C=c)
    clf.fit(X_train, y_train)
    coefs_.append(clf.coef_.ravel().copy())
    scores.append(clf.score(X_test, y_test))

coefs_ = np.array(coefs_)
# print(coefs_.shape)
coef_indices_to_use = np.argpartition(np.abs(coefs_[-1,:]), -1 * n_coefs)[-1 * n_coefs:]
# print(coef_indices_to_use)

figp, (axp, axps) = plt.subplots(nrows=2, figsize = (6,10))
for i in coef_indices_to_use:
    try:
        axp.plot(np.log10(cs), coefs_[:,i], label = names[i]) #marker = 'o')
    except IndexError:
        axp.plot(np.log10(cs), coefs_[:,i], label = str(i)) #marker = 'o')

axps.plot(np.log10(cs), scores)

axp.legend()
figp.tight_layout()
figp.savefig('./figs/p4_LogReg.png')




#
# # classifier using lasso regression with CV
# cs = l1_min_c(X_train, y_train, loss='log') * np.logspace(0, 5, 16)
#
# clf_lr = linear_model.LogisticRegressionCV(
#                                         cv=5,
#                                         Cs=cs,
#                                         penalty='l1',
#                                         solver='saga',
#                                         # tol=1e-6,
#                                         max_iter=int(1e6),
#                                         # n_jobs=-1,
#                                         fit_intercept=False
#                                         )
# print('Training')
#
# # for this we need to standardize the data
#
# clf_lr.fit(X_train, y_train)
#
# coef_paths = clf_lr.coefs_paths_[1]
# cs = clf_lr.Cs_
#
# scores = clf_lr.scores_[1]
#
# best_score_ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
# best_fold_ind = best_score_ind[0]
# best_c_ind = best_score_ind[1]
#
# best_score = scores[best_score_ind]
#
# names = features_to_train.copy()
# # names.append('w0') # added to account for intercept fitting in clf
#
#
# fig, axes = plt.subplots(nrows=2, figsize = (6,10))
# ax = axes.ravel()
# for j, p in enumerate(coef_paths[best_fold_ind].T):
#     try:
#         ax[0].plot(np.log10(cs), p, label=names[j])
#     except IndexError:
#         ax[0].plot(np.log10(cs), p, label='1/{}'.format(names[j - len(names)]))
#
#     # ax[0].plot(cs, p, label=names[j] if fi == 0 else None)
# # for fi, ps in enumerate(coef_paths):
# #     for j, p in enumerate(ps.T):
# #         if fi == best_fold_ind:
# #             ax[0].plot(cs, p, label=names[j])
# #         # ax[0].plot(cs, p, label=names[j] if fi == 0 else None)
#
# for fi, foldscore in enumerate(scores):
#     ax[1].plot(np.log10(cs), foldscore, label='fold {}'.format(fi+1))
#
#
# ax[0].axvline(np.log10(cs[best_c_ind]), ls='--',c='r',lw=1)
# ax[1].axvline(np.log10(cs[best_c_ind]), ls='--',c='r',lw=1)
# ax[0].set_ylabel('Coefficient value')
# ax[1].set_ylabel('True Positive Rate (zoomed)')
# ax[1].set_xlabel('log10(C value)')
#
#
# ax[0].set_title('Coefficient profiles for best performing fold ({})'.format(best_fold_ind+1))
# ax[0].legend(loc='lower right')
# ax[1].set_title('CV Classification rates. Max: {:.3f} at c={:.2f} '.
#                 format(best_score,np.log10(cs[best_c_ind])))
# ax[1].legend(loc='upper left')\
#
# # ax[1].set_ylim(0,1)
#
# # ax.legend()
# fig.tight_layout()
# fig.savefig('./figs/p4_LogRegCV_now0.pdf')
#


# # classifier with xgboost
#
# dtrain = xgb.DMatrix(df_train.drop(columns=['ID','No-show']), label=df_train['No-show'])
# dtest = xgb.DMatrix(df_test.drop(columns=['ID','No-show']), label=df_test['No-show'])
#
# num_round = 10
#
# param = {
#     # 'max_depth': 2,
#     # 'eta': 1,
#     'verbosity': 0,
#     'objective': 'binary:logitraw'
#     # 'objective': 'binary:logistic'
# }
#
# bst = xgb.train(param, dtrain, num_boost_round=num_round, )
# # bst = xgb.train(param, dtrain, num_round, evallist)
#
# bst.dump_model('dump.raw.txt')
#
# yprobs = bst.predict(dtest)
#
# y_pred = yprobs > 0.5
#
# # print(y_pred)
#
# y_val = np.array(np.equal(y_pred, df_test['No-show'].values), dtype=int)
# print(np.sum(y_val))
# print(len(y_val))
#
# print('TPR: {}'.format(np.sum(y_val) / len(y_val)))
#
# fig, ax = plt.subplots()
#
# bin_centers, counts, unc, mask = lm.bin_data(yprobs, N_bins=25)
#
# ax.errorbar(bin_centers[mask], counts[mask], yerr=unc[mask], )
#
# fig.tight_layout()
# plt.show()
