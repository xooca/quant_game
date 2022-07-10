from sklearn.pipeline import Pipeline
import pickle
import logging
import data.data_engine as de
import pandas as pd
import numpy as np
#import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf

def run_pipeline(pipe_list,df,pipeinfo_loc,data_loc,load_previous = True):
    pipe_list_save = [col for col in pipe_list]
    if load_previous:
        try:
            with open(pipeinfo_loc, 'rb') as handle:
                pipe_list = pickle.load(handle)
            logging.info(f"Previous pipeline loaded from location {pipeinfo_loc}. Length of pipeline is {len(pipe_list)}")
            df = pd.read_csv(data_loc,parse_dates=True,index_col='Unnamed: 0')
            logging.info(f"Previous data loaded from location {data_loc}. Shape of the data is {df.shape}")
        except Exception as e1:
            logging.info(f"File {pipeinfo_loc} is not loaded because of error : {e1}")
    for i, pipe in enumerate(pipe_list,1):
        logging.info('#'*100)
        logging.info(f"Pipeline {i} started. Shape of the data is {df.shape}")
        logging.info(pipe)
        df = pipe.fit_transform(df)
        pipe_list_save.remove(pipe)
        df.to_csv(data_loc)
        with open(pipeinfo_loc, 'wb') as handle:
            pickle.dump(pipe_list_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Pipeline {i} completed. Shape of the data is {df.shape}")
    return df

def initialize_config(overrides,version_base=None, config_path="../config"):
    initialize(version_base=version_base, config_path=config_path)
    dc=compose(overrides= overrides)
    return dc