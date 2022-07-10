import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import importlib
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
GlobalHydra.instance().clear()
#importlib.import_module('data.data_pipelines.generic_pipeline')
import data.data_pipelines as dd
import data.data_utils as du

base_df = pd.read_csv(dd.dc.data.paths.input_path)
all_func = dd.__dict__
for datapipeline,subdatapipeline in dd.dc.data.datapipeline.items():
    all_pipe = [all_func[pipe] for pipe in subdatapipeline]
    print(all_pipe)
    df = du.run_pipeline(all_pipe,base_df,dd.dc.data.paths.pipeinfo,f"dd.dc.data.paths.base_data_loc{datapipeline}.csv",load_previous=False)