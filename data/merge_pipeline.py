import pandas as pd
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
from sklearn.model_selection import train_test_split
GlobalHydra.instance().clear()
#importlib.import_module('data.data_pipelines.generic_pipeline')
import data.data_pipelines as dd

master_pipeline_path = f"{dd.dc.data.paths.base_data_loc}{dd.dc.data.datapipeline.master_pipeline}.csv"
master_df = pd.read_csv(master_pipeline_path,parse_dates=True,index_col='Unnamed: 0')
master_df_col = [col for col in master_df.columns.tolist() if col not in dd.dc.data.datapipeline.pipeline1_exclude_cols]
master_df = master_df[master_df_col]

for pipeline in dd.dc.data.datapipeline.merge_pipeline_to_master:
    tmppath = f"{dd.dc.data.paths.base_data_loc}{pipeline}.csv"
    print(tmppath)
    tmp_exclude_cols = dd.dc.data.datapipeline[f"{pipeline}_exclude_cols"]
    tmpdf = pd.read_csv(tmppath,parse_dates=True,index_col='Unnamed: 0')
    tmp_cols = [col for col in tmpdf.columns.tolist() if col not in tmp_exclude_cols]
    tmpdf = tmpdf[tmp_cols]
    master_df = pd.merge(master_df,tmpdf, how='inner', left_index=True, right_index=True)

if dd.dc.data.data_split.test_percent is None:
    test_size = 0.2
else:
    test_size = dd.dc.data.data_split.test_percent

if dd.dc.data.data_split.stratify_col is not None:
    train, test = train_test_split(master_df, test_size=test_size,stratify=master_df[dd.dc.data.data_split.stratify_col])
    test.to_csv(dd.dc.data.paths.test_save_path)
    if dd.dc.data.data_split.valid_percent is not None:
        train, valid = train_test_split(train, test_size=dd.dc.data.data_split.valid_percent,stratify=master_df[dd.dc.data.data_split.stratify_col])
        train.to_csv(dd.dc.data.paths.train_save_path)
        valid.to_csv(dd.dc.data.paths.valid_save_path)
else:
    train, test = train_test_split(master_df, test_size=test_size)
    test.to_csv(dd.dc.data.paths.test_save_path)
    if dd.dc.data.data_split.valid_percent is not None:
        train, valid = train_test_split(train, test_size=dd.dc.data.data_split.valid_percent)
        train.to_csv(dd.dc.data.paths.train_save_path)
        valid.to_csv(dd.dc.data.paths.valid_save_path)
