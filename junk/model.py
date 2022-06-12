from pycaret.classification import *
import pandas as pd
base_df3_15 = pd.read_csv(r'C:\Users\8prab\Google Drive\Work\trading\database_df3_15.csv')
base_df3_15.index = pd.DatetimeIndex(base_df3_15.index)
base_df3_15 = base_df3_15.sort_index()
base_df3_15 = base_df3_15[~base_df3_15.index.duplicated(keep='first')]
print(base_df3_15)
print(base_df3_15.dtypes)
exp_mclf101 = setup(data = base_df3_15, target = 'label_-15_1min_close', session_id=123) 
best = compare_models()

