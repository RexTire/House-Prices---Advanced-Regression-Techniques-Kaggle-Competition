import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')


df_nan_index_list = df.isnull().sum().values.tolist()

no_of_nan = []
index_of_nan = []
name_of_nan_col = []

# appending values
for i in df_nan_index_list:
    if (i != 0):
        no_of_nan.append(i)

# appending index of df
for i in range(len(df_nan_index_list)):
    if (df_nan_index_list[i] != 0):
        index_of_nan.append(i)

# appending the column name
for i in index_of_nan:
    name_of_nan_col.append(df.columns[i])

nan_arr = np.array([index_of_nan, no_of_nan, name_of_nan_col])

nan_arr = nan_arr.T


nan_df = pd.DataFrame(nan_arr, columns=['index', 'no of nan values', 'column name'])
nan_df['% of nan'] = nan_df['no of nan values'].astype(int) / len(df) * 100

nan_df.to_csv('nan_df.csv', index=False)
