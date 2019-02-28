import pandas as pd

# read data
df_bg_train = pd.read_csv('../wk5/BDT_background_train.txt')
df_sg_train = pd.read_csv('../wk5/BDT_signal_train.txt')

df_bg_test = pd.read_csv('../wk5/BDT_background_test.txt')
df_sg_test = pd.read_csv('../wk5/BDT_signal_test.txt')

# assign labels
df_bg_train['label'] = 0
df_bg_test ['label'] = 0
df_sg_train['label'] = 1
df_sg_test ['label'] = 1

print(df_sg_train.dtypes)

