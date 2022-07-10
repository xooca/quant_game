import pandas as pd
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
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
