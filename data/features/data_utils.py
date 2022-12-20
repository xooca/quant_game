from sklearn.pipeline import Pipeline
import data.features.data_engine as de
import pickle
import logging
import requests
import omegaconf
#import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf
import gc
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
import importlib
from sklearn.model_selection import train_test_split
import zipfile,fnmatch,os
from pathlib import Path
import string
import re
import gc
from config.common.config import Config,DefineConfig
from datetime import datetime

warnings.filterwarnings('ignore')

os.getcwd()
pd.options.display.max_columns = None

def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=["float64"]).columns)
    if len(list_of_columns)>=1:        
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    else:
        print("no columns to downcast")
    gc.collect()
    return df

def downcast_df_int_columns(df):
    list_of_columns = list(df.select_dtypes(include=["int32", "int64"]).columns)
    if len(list_of_columns)>=1:        
        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    else:
        print("no columns to downcast")
    gc.collect()
    return df

def reduce_mem_usage_v1(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    for var in df.columns.tolist():
        if df[var].dtype != object:
            maxi = df[var].max()
            if maxi < 255:
                df[var] = df[var].astype(np.uint8)
                print(var,"converted to uint8")
            elif maxi < 65535:
                df[var] = df[var].astype(np.uint16)
                print(var,"converted to uint16")
            elif maxi < 4294967295:
                df[var] = df[var].astype(np.uint32)
                print(var,"converted to uint32")
            else:
                df[var] = df[var].astype(np.uint64)
                print(var,"converted to uint64")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            if str(col_type)[:3] == 'int':
                df[col] = df[col].astype(np.int32)
            if str(col_type)[:5] == 'float':
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def reduce_mem_usage_v2(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def remove_specialchars(text):
    text_re = re.escape(string.punctuation)
    return re.sub(r'['+text_re+']', '',text)

def nullcolumns(df):
    t = pd.DataFrame(df[df.columns[df.isnull().any()]].isnull().sum()).reset_index()
    t.columns = ['colname','nullcnt']
    t = t.sort_values(by='nullcnt',ascending=False)
    return t

def checknans(df,threshold=100):
    nan_cols =[]
    for col in df.columns.tolist() :
        if sum(np.isnan(df[col])) > threshold:
            print(f"{col}.... {sum(np.isnan(df.train[col]))}")
            nan_cols.append(col)
    return nan_cols

def initialize_config(overrides,version_base=None, config_path="../config"):
    initialize(config_path=config_path)
    dc=compose(overrides= overrides)
    return dc

def print_log(log,using_print='print'):
    if using_print=='print':
        print(log)
    else:
        logging.info(log)

def check_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def convert_df_to_timeseries(df):
    df['date_time'] = df['date'].astype(str) + ' ' + df['time']
    df = df.sort_values(by='date_time')
    df.index = df['date_time']
    df = df[['open','high','low','close']]
    return df

def load_object(object_path):
    with open(object_path, 'rb') as handle:
        return_obj = pickle.load(handle)
        print_log(f"Object loaded")
    return return_obj

def save_object(object_path,obj):
    with open(object_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print_log(f"Object saved at location {object_path}")

class initial_data_setup(DefineConfig):
    def __init__(self,master_config_path):
        DefineConfig.__init__(self,master_config_path)

    def unzip_folders(self):
        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.zip_file_pattern):
                f_name = os.path.join(root, filename)
                try:
                    if zipfile.is_zipfile(f_name):
                        n_file = os.path.join(root, os.path.splitext(filename)[0])
                        zipfile.ZipFile(f_name).extractall(n_file)
                        print_log(f"File saved at location {n_file}",self.using_print)
                        os.remove(f_name)
                        print_log(f"File {f_name} removed",self.using_print)
                    else:
                        print_log(f"File {f_name} is not unzipped",self.using_print)
                except Exception as e1:
                    print_log(f"File {f_name} is not unzipped",self.using_print)
                    print_log(f"Error encountered is {e1}",self.using_print)

    def create_dataset(self,reload_all = True):
        files_list = []
        bad_files = []
        files_processed = []
        base_df = pd.DataFrame(columns = self.initial_columns)
        print_log(f'Source data path is { self.source_data}')
        if not os.path.exists(self.source_data):
            os.makedirs(self.source_data)
            print_log(f'Created folder {self.source_data}',self.using_print)

        already_loaded_file_name = f'{self.source_data}already_loaded_files.pickle'
        print_log(f'Data save path is { self.raw_data_save_path}')
        print_log(f'File with already loaded files is {already_loaded_file_name}')
        try:
            with open(already_loaded_file_name, 'rb') as handle:
                already_loaded_files = pickle.load(handle)
                already_loaded_files = [Path(col) for col in already_loaded_files]
                print_log(f"Total files already saved {len(already_loaded_files)}",self.using_print)
        except Exception as e1:
            print_log(f"File {already_loaded_file_name} is not loaded because of error : {e1}",self.using_print)
            already_loaded_files = []
        print_log(f"Raw data root path is {self.raw_data_input_path}",self.using_print)
        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.data_pattern):
                f_name = Path(os.path.join(root, filename))
                files_list.append(f_name)

        files_to_be_loaded = [f for f in files_list if f not in already_loaded_files]
        files_to_be_loaded = list(dict.fromkeys(files_to_be_loaded))
        files_list = list(dict.fromkeys(files_list))
        print_log(f"Total files detected {len(files_list)}",self.using_print)
        print_log(f"Total new files detected {len(files_to_be_loaded)}",self.using_print)
        
        try:
            base_df = pd.read_csv(self.raw_data_save_path)
        except Exception as e1:
            print_log(f"Error while loading dataframe from { self.raw_data_save_path} because of error : {e1}")
            base_df = pd.DataFrame(columns = self.ohlc_column)
            files_to_be_loaded = files_list
        if len(base_df) == 0 or reload_all:
            files_to_be_loaded = files_list
            print_log(f"We are going to reload all the data",self.using_print)
        print_log(f"Number of files to be loaded {len(files_to_be_loaded)}",self.using_print)
        base_df_st_shape = base_df.shape
        files_to_be_loaded = sorted(files_to_be_loaded)
        for i,f_name in enumerate(files_to_be_loaded,1):
            f_name = os.path.join(root, f_name)
            try:
                tmp_df = pd.read_csv(f_name,header=None)
                tmp_df = tmp_df.loc[:,0:6]
                tmp_df.columns = self.initial_columns
                tmp_df = convert_df_to_timeseries(tmp_df)
                base_df = pd.concat([base_df,tmp_df],axis=0)
                print_log(f"Data shape after loading file {f_name} is {base_df.shape}",self.using_print)
                print_log(f"Files left to be loaded {len(files_to_be_loaded)-i}",self.using_print)
                already_loaded_files.append(f_name)
            except Exception as e1:
                bad_files.append(f_name)
                print_log(f"File {f_name} is not loaded because of error : {e1}",self.using_print)
        with open(already_loaded_file_name, 'wb') as handle:
            pickle.dump(already_loaded_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print_log(f"Shape of the dataframe before duplicate drop is {base_df.shape}",self.using_print)
        base_df = base_df.drop_duplicates()
        print_log(f"Shape of the dataframe after duplicate drop is {base_df.shape}",self.using_print)
        #if base_df_st_shape != base_df.shape:
        base_df = base_df.sort_index()
        base_df.to_csv( self.raw_data_save_path)
        print_log(f"Saving dataframe to location { self.raw_data_save_path}",self.using_print)
        return base_df
class execute_data_pipeline(DefineConfig):
    def __init__(self,master_config_path):
        DefineConfig.__init__(self,master_config_path)
        check_and_create_dir(self.base_data_loc)
        print_log(f"Feature spec file is {self.train_datapipeline_spec}",self.using_print)
        feature_spec = importlib.import_module(f"{self.train_datapipeline_spec}")
        self.feature_pipeline = feature_spec.pipelines(self.config)
        self.custom_parameter_definition()
        self.base_df = pd.DataFrame()
    
    def custom_parameter_definition(self):
        self.datapipeline = self.config.data.datapipeline

    def run_individual_pipeline(self,datapipeline,pipe_name,pipe):
        print_log(f"*************************** pipe {pipe_name} started.**********************************",self.using_print)
       
        print_log(f"Below is the descripton of pipe",self.using_print)
        print_log(pipe,self.using_print)
        pipe_dir = f"{self.base_data_loc}saved_pipeline/{datapipeline}"
        pipe_data_dir = f"{self.base_data_loc}saved_data_pipeline/{datapipeline}/"
        check_and_create_dir(pipe_dir)
        check_and_create_dir(pipe_data_dir)
        pipe_file = f"{pipe_dir}/pipe_{pipe_name}.pkl"
        pipe_data_location = f"{pipe_data_dir}{pipe_name}.csv"
        if self.train_load_previous:
            if os.path.exists(pipe_file):
                print_log(f"{pipe_file} already exists. skipping {pipe_name}",self.using_print)
                return
            else:
                self.train_load_previous = False
        print_log(f"Pipe {pipe_name} started. Shape of the data is {self.base_df.shape}",self.using_print)
        df = pipe.fit_transform(self.base_df)
        df.to_csv(pipe_data_location)
        with open(pipe_file, 'wb') as handle:
            pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print_log(f"{pipe_file} saved",self.using_print)
        print_log(f"pipe {pipe_name} completed. Save at location {pipe_file}",self.using_print)
        print_log(f"After completion of pipe {pipe_name}, shape of the data is {df.shape}",self.using_print)
        del df
        gc.enable()
        gc.collect()

    def run_pipeline(self,datapipeline,subdatapipeline,all_func,df):
        print_log(f"*************************** Pipeline {datapipeline} started.**********************************",self.using_print)
        if datapipeline in all_func:
            all_pipe = [self.feature_pipeline.__dict__[pipe] for pipe in subdatapipeline if pipe in self.feature_pipeline.__dict__.keys()]
            name_pipe_list = [pipe for pipe in subdatapipeline if pipe in self.feature_pipeline.__dict__.keys()]
            if len(all_pipe)>0:
                print_log(all_pipe,self.using_print)
                print_log(f"{self.base_data_loc}{datapipeline}.pkl",self.using_print)
                print_log(f"{self.base_data_loc}{datapipeline}.csv",self.using_print)
                
                flag_file = f"{self.base_data_loc}{datapipeline}_flag.pkl"
                pipe_flag = 'completed'

                if self.train_load_previous:
                    if os.path.exists(flag_file):
                        print_log(f"{flag_file} already exists. skipping {datapipeline}",self.using_print)
                        return
                for pipe_name, pipe in zip(name_pipe_list,all_pipe):
                    self.run_individual_pipeline(datapipeline,pipe_name,pipe)
                with open(flag_file, 'wb') as handle:
                    pickle.dump(pipe_flag, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def run_initial_pipeline(self,initial_pipeline_save_path,df=None,save_df=True):
        if df is None:
            initial_df = pd.read_csv(self.input_path,parse_dates=True,index_col='Unnamed: 0')
        else:
            initial_df = df
        self.feature_pipeline.pipeline_definitions()
        initial_pipelines = [self.feature_pipeline.__dict__[pipe] for pipe in self.initial_pipeline if pipe in self.feature_pipeline.__dict__.keys()]
        check_and_create_dir(os.path.dirname(initial_pipeline_save_path))
        for initial_pipeline in initial_pipelines:
            initial_df = initial_pipeline.fit_transform(initial_df)
        if save_df:
            initial_df.to_csv(initial_pipeline_save_path)
        print_log(f"Initial dataframe size is {initial_df.shape}",self.using_print)
        
    def run_main_pipeline(self,initial_pipeline_save_path):
        self.base_df = pd.read_csv(initial_pipeline_save_path,parse_dates=True,index_col='Unnamed: 0')
        print_log(f"Shape of data is {self.base_df.shape}",self.using_print)
        self.feature_pipeline.pipeline_definitions()
        all_func = self.config._content['data']['datapipeline']
        final_pipeline = {}
        pipelines_dict = dict(self.datapipeline)
        final_pipeline_path = f"{self.base_data_loc}final_pipeline.pkl"
        print_log(f"Size of dataframe before transformation {self.base_df.shape}",self.using_print)
        for datapipeline,subdatapipeline in pipelines_dict.items():
            self.run_pipeline(datapipeline,subdatapipeline,all_func,self.base_df)
            final_pipeline.update({datapipeline:subdatapipeline})
            print_log(f"*************************** Pipeline {datapipeline} completed.**********************************",self.using_print)
        with open(final_pipeline_path, 'wb') as handle:
            pickle.dump(final_pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print_log(f"Final pipeline saved to {final_pipeline_path}",self.using_print)
        print_log(f"*************************** PIPELINE EXECUTION COMPLETED**********************************",self.using_print)
        
    def load_and_run_individual_pipeline(self,pipe_location,datapipeline,pipe_name,pipe_data_location):
        pipe_dir = f"{pipe_location}saved_pipeline/{datapipeline}"
        pipe_file = f"{pipe_dir}/pipe_{pipe_name}.pkl"
        if not os.path.exists(pipe_file):
            print_log(f"{pipe_file} does not exists. skipping {pipe_name}",self.using_print)
            return
        with open(pipe_file, 'rb') as handle:
            pipe = pickle.load(handle)
        self.base_df = pipe.transform(self.base_df)
        #self.base_df.to_csv(pipe_data_location)
        print_log(f"Data saved at location {pipe_data_location}",self.using_print)
            
    def load_and_run_pipeline(self,pipe_location,datapipeline,subdatapipeline):
        check_and_create_dir(self.predict_base_path)
        pipe_data_location = f"{self.predict_base_path}{datapipeline}.csv"
        for pipe_name in subdatapipeline:
            self.load_and_run_individual_pipeline(pipe_location,datapipeline,pipe_name,pipe_data_location)

    def load_run_main_pipeline(self,initial_pipeline_save_path):
        self.base_df = pd.read_csv(initial_pipeline_save_path,parse_dates=True,index_col='Unnamed: 0')
        final_pipeline_path = f"{self.base_data_loc}final_pipeline.pkl"
        with open(final_pipeline_path, 'rb') as handle:
            pipelines_dict = pickle.load(handle)
        print_log(f"Below is the description of full pipeline :",self.using_print)
        print_log(f"{pipelines_dict}",self.using_print)
        final_predict_path = f"{self.predict_base_path}final_predict_df.csv"
        for datapipeline,subdatapipeline in pipelines_dict.items():
            self.load_and_run_pipeline(self.base_data_loc,datapipeline,subdatapipeline)
        self.base_df.to_csv(final_predict_path)
            
    def run_prediction_pipeline(self,initial_pipeline_save_path):
        self.run_initial_pipeline(initial_pipeline_save_path)
        self.load_run_main_pipeline(initial_pipeline_save_path)    
     
    def run_training_pipeline(self,initial_pipeline_save_path):
        self.run_initial_pipeline(initial_pipeline_save_path)
        self.run_main_pipeline(initial_pipeline_save_path)
    
    def execute_pipeline(self,pipeline_type='training'):
        if pipeline_type=='training':
            self.run_training_pipeline(self.train_initial_file_path)
        else:
            self.run_prediction_pipeline(self.predict_initial_file_path)
            
    def time_based_split(self,master_df):
        train = master_df[master_df.index.to_series().between(self.train_date_range_ll, self.train_date_range_ul)]
        print_log(f"Shape of train df is : {train.shape}",self.using_print)
        test = master_df[master_df.index.to_series().between(self.test_date_range_ll, self.test_date_range_ul)]
        print_log(f"Shape of test df is : {test.shape}",self.using_print)
        valid = master_df[master_df.index.to_series().between(self.valid_date_range_ll, self.valid_date_range_ul)]
        print_log(f"Shape of valid df is : {valid.shape}",self.using_print)
        del master_df
        gc.enable()
        gc.collect()
        train.to_csv(self.train_training_data_output_path)
        print_log(f"Train data saved at location {self.train_training_data_output_path}",self.using_print)
        valid.to_csv(self.train_validation_data_output_path)
        print_log(f"Valid data saved at location {self.train_validation_data_output_path}",self.using_print)
        test.to_csv(self.train_testing_data_output_path)
        print_log(f"Test data saved at location {self.train_testing_data_output_path}",self.using_print)
        

    def stratified_split(self,train):
        print_log(f"Stratify column is {self.train_stratify_col}",self.using_print)
        train, test = train_test_split(train, test_size=self.train_testing_percent,stratify=train[self.train_stratify_col])
        test.to_csv(self.train_testing_data_output_path)
        print_log(f"Shape of test df is : {test.shape}",self.using_print)
        print_log(f"Test data saved at location {self.train_testing_data_output_path}",self.using_print)
        if self.train_validation_percent is not None:
            print_log(f"Valid split size is {self.train_validation_percent}",self.using_print)
            train, valid = train_test_split(train, test_size=self.train_validation_percent,stratify=train[self.train_stratify_col])
            train.to_csv(self.train_training_data_output_path)
            print_log(f"Shape of train df is : {train.shape}",self.using_print)
            print_log(f"Train data saved at location {self.train_training_data_output_path}",self.using_print)
            valid.to_csv(self.train_validation_data_output_path)
            print_log(f"Shape of valid df is : {valid.shape}",self.using_print)
            print_log(f"Validation data saved at location {self.train_validation_data_output_path}",self.using_print)
    
    def random_split(self,train):
        train, test = train_test_split(train, test_size=self.train_testing_percent)
        test.to_csv(self.train_testing_data_output_path)
        print_log(f"Shape of test df is : {test.shape}",self.using_print)
        if self.train_validation_percent is not None:
            train, valid = train_test_split(train, test_size=self.train_validation_percent)
            train.to_csv(self.train_training_data_output_path)
            print_log(f"Shape of train df is : {train.shape}",self.using_print)
            print_log(f"Train data saved at location {self.train_training_data_output_path}",self.using_print)
            valid.to_csv(self.train_validation_data_output_path)
            print_log(f"Shape of valid df is : {valid.shape}",self.using_print)
            print_log(f"Validation data saved at location {self.train_validation_data_output_path}",self.using_print)

    def load_all_data_from_dir(self,path,tmp_exclude_cols=[]):
        files = os.listdir(path)
        files = [f for f in files if f.split(".")[-1]=='csv']
        print_log(f"\tNumber of files found : {len(files)}",self.using_print)
        list_df = []
        for f in files:
            f_path = f"{path}{f}"
            tmpdf = pd.read_csv(f_path,parse_dates=True,index_col='Unnamed: 0')
            print_log(f"\tFile loaded from path : {f_path}",self.using_print)
            print_log(f"\tShape of the df is : {tmpdf.shape}",self.using_print)
            tmp_cols = [col for col in tmpdf.columns.tolist() if col not in tmp_exclude_cols]
            print_log(f"\tColumns excluded : {tmp_exclude_cols}",self.using_print)
            tmpdf = tmpdf[tmp_cols]
            list_df.append(tmpdf)
        return pd.concat(list_df,axis=1)
    
    def merge_data(self):
        master_pipeline_path = f"{self.base_data_loc}saved_data_pipeline/{self.master_pipeline}/"
        print_log(f"Started reading master data from path : {master_pipeline_path}",self.using_print)
        master_df = self.load_all_data_from_dir(master_pipeline_path,tmp_exclude_cols=self.master_pipeline_exclude_cols)
        print_log(f"Completed reading master data from path : {master_pipeline_path}",self.using_print)
        print_log(f"Shape of master df before merge {master_df.shape}",self.using_print)
        for pipeline in self.merge_pipeline_to_master:
            pipe_data_dir = f"{self.base_data_loc}saved_data_pipeline/{pipeline}/"
            tmp_exclude_cols = eval(f"self.{pipeline}_exclude_cols")
            tmpdf = self.load_all_data_from_dir(pipe_data_dir,tmp_exclude_cols)
            print_log(f"Shape of df in {pipeline} is {tmpdf.shape}",self.using_print)
            tmp_cols = [col for col in tmpdf.columns.tolist() if col not in tmp_exclude_cols]
            tmpdf = tmpdf[tmp_cols]
            master_df = pd.merge(master_df,tmpdf, how='inner', left_index=True, right_index=True)
            del tmpdf
            print_log(f"Shape of data after merging {pipeline} is : {master_df.shape}",self.using_print)
            gc.enable()
            gc.collect()
        return master_df
        
    def merge_pipeline(self,split_flag=True):
        master_df =self.merge_data()
        filter_pipe = Pipeline([('save_fd', de.FilterData(start_date=self.save_start_date,end_date=self.save_end_date)),])
        master_df = reduce_mem_usage(master_df)
        master_df = filter_pipe.fit_transform(master_df)
        print_log(f"Shape of data after filtering is : {master_df.shape}",self.using_print)
        if split_flag:
            if self.splits_type == 'normal':
                print_log(f"Started splitting data into train, test and validation",self.using_print)
                print_log(f"Test split size is {self.train_testing_percent}",self.using_print)
                if self.train_stratify_col is not None or self.train_stratify_col != 'None':
                    print_log(f"Split type is stratified split",self.using_print)
                    self.stratified_split(master_df)
                else:
                    print_log(f"Split type is random split",self.using_print)
                    self.random_split(master_df)
            elif self.splits_type == 'time_based':
                print_log(f"Split type is time based split",self.using_print)
                self.time_based_split(master_df)
            else:
                print_log(f"split type {self.splits_type} is not correct. Please mention split type as normal or time_based",self.using_print)
        else:
            master_df.to_csv(self.train_training_data_output_path)


class execute_prediction_data_pipeline(DefineConfig):
    def __init__(self,master_config_path):
        DefineConfig.__init__(self,master_config_path)
        check_and_create_dir(self.base_data_loc)
        print_log(f"Feature spec file is {self.train_datapipeline_spec}",self.using_print)
        feature_spec = importlib.import_module(f"{self.train_datapipeline_spec}")
        self.feature_pipeline = feature_spec.pipelines(self.config)
        self.custom_parameter_definition()
        self.base_df = pd.DataFrame()
    
    def custom_parameter_definition(self):
        self.datapipeline = self.config.data.datapipeline
            
    def load_and_run_individual_pipeline(self,pipe_location,datapipeline,pipe_name,pipe_data_location):
        pipe_dir = f"{pipe_location}saved_pipeline/{datapipeline}"
        pipe_file = f"{pipe_dir}/pipe_{pipe_name}.pkl"
        if not os.path.exists(pipe_file):
            print_log(f"{pipe_file} does not exists. skipping {pipe_name}",self.using_print)
            return
        with open(pipe_file, 'rb') as handle:
            pipe = pickle.load(handle)
        self.base_df = pipe.transform(self.base_df)
        #self.base_df.to_csv(pipe_data_location)
        print_log(f"Data saved at location {pipe_data_location}",self.using_print)
            
    def load_and_run_pipeline(self,pipe_location,datapipeline,subdatapipeline):
        check_and_create_dir(self.predict_base_path)
        pipe_data_location = f"{self.predict_base_path}{datapipeline}.csv"
        for pipe_name in subdatapipeline:
            self.load_and_run_individual_pipeline(pipe_location,datapipeline,pipe_name,pipe_data_location)

    def load_run_prediction_pipeline(self,df,if_save=False):
        self.base_df = df
        final_pipeline_path = f"{self.base_data_loc}final_pipeline.pkl"
        with open(final_pipeline_path, 'rb') as handle:
            pipelines_dict = pickle.load(handle)
        print_log(f"Below is the description of full pipeline :",self.using_print)
        print_log(f"{pipelines_dict}",self.using_print)
        final_predict_path = f"{self.predict_base_path}final_predict_df.csv"
        for datapipeline,subdatapipeline in pipelines_dict.items():
            self.load_and_run_pipeline(self.base_data_loc,datapipeline,subdatapipeline)
        if if_save:
            self.base_df.to_csv(final_predict_path) 
     
class read_data_api:
    def __init__(self,master_config):
        master_config = dict(master_config['master']['model'])
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = initialize_config(**master_config)
        self.using_print = True if self.config.data.generic.verbose_type == 'print' else False
        print(self.config)
        self.base_url = self.config.common.endpoint.base_url
        self.endpoint_details = self.config.common.endpoint_details
        self.read_action = self.config.common.endpoint.read_action

    def create_url(self,url=None):
        if url is None:
            endpoint_details = dict(self.endpoint_details)
            sub_url = ''
            for i,j in endpoint_details.items():
                if j != 'None':
                    sub_url += f"{i}={j}&"
            self.url = f"{self.base_url}{sub_url}"
            self.url = self.url[:-1]
        else:
            self.url = url
        print_log(f"API URL is {self.url}",self.using_print)

    def call_api(self,url=None,read_action=None):
        self.create_url(url)
        if read_action is None:
            if self.read_action == 'json':
                r = requests.get(self.url)
                data = r.json()
            elif self.read_action == 'csv':
                data = pd.read_csv(self.url)
            else:
                print_log(f"Invalid read action: {self.read_action}",self.using_print)
                data = None
        return data
class datacleaner:
    def __init__(self, df, targetcol, id_cols=None, cat_threshold=100):
        self.df_train = df
        self.target = targetcol
        self.id = id_cols
        self.dfcolumns = self.df_train.columns.tolist()
        self.dfcolumns_nottarget = [
            col for col in self.dfcolumns if col != self.target]
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.num_cols = self.df_train.select_dtypes(
            include=numerics).columns.tolist()
        self.num_cols = [
            col for col in self.num_cols if col not in [self.target, self.id]]
        self.non_num_cols = [
            col for col in self.dfcolumns if col not in self.num_cols + [self.target, self.id]]
        self.rejectcols = []
        self.retainedcols = []
        self.catcols = {}
        self.noncatcols = {}
        self.catcols_list = []
        self.noncatcols_list = []
        self.hightarge_corr_col = []
        self.threshold = cat_threshold

    def normalize_column_name(self,col_name):
        col_name = str(col_name)
        col_name = col_name.lower()
        col_name = col_name.strip()     
        col_name = col_name.replace(' ', '_')         
        col_name = col_name.replace(r"[^a-zA-Z\d\_]+", "")
        return col_name

    def normalize_metadata(self,tmpdf):
        self.df_train = tmpdf
        self.target = self.normalize_column_name(self.target)
        self.id = self.normalize_column_name(self.id)
        self.target = self.normalize_column_name(self.target)
        self.dfcolumns = [self.normalize_column_name(col) for col in self.dfcolumns]
        self.dfcolumns_nottarget = [self.normalize_column_name(col) for col in self.dfcolumns_nottarget]
        self.num_cols = [self.normalize_column_name(col) for col in self.num_cols]
        self.non_num_cols = [self.normalize_column_name(col) for col in self.non_num_cols]

    def clean_column_name(self):
        def clean_column_name_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    tmpdf.columns = [str(x).lower() for x in tmpdf.columns.tolist()]
                    tmpdf.columns = tmpdf.columns.str.strip()     
                    tmpdf.columns = tmpdf.columns.str.replace(' ', '_')         
                    tmpdf.columns = tmpdf.columns.str.replace(r"[^a-zA-Z\d\_]+", "")  
                    self.normalize_metadata(tmpdf)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return clean_column_name_lvl

    def reject_null_cols(self, null_threshold):
        def reject_null_cols_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in tmpdf:
                        null_count = sum(tmpdf[col].astype(str).isnull())
                        if null_count > 0:
                            percent_val = null_count/tmpdf[col].shape[0]
                            if percent_val > null_threshold:
                                self.rejectcols.append(col)
                    self.retainedcols = [
                        col for col in tmpdf.columns.tolist() if col not in self.rejectcols]
                    print(
                        f"INFO : {str(datetime.now())} : Number of rejected columns {len(self.rejectcols)}")
                    print(
                        f"INFO : {str(datetime.now())} : Number of retained columns {len(self.retainedcols)}")
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper

        return reject_null_cols_lvl

    def standardize_stratified(self, auto_standard=True, includestandcols=[],for_columns = []):
        def standardize_stratified_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    if len(for_columns) ==0:
                        if auto_standard:
                            stand_cols = list(
                                set(self.num_cols + includestandcols))
                        else:
                            stand_cols = self.num_cols
                    else:
                        stand_cols = for_columns.copy()
                    for col in tmpdf:
                        if col in stand_cols:
                            tmpdf[col] = tmpdf[col].astype(np.float)
                            tmpdf[col] = tmpdf[col].replace(np.inf, 0.0)
                            if tmpdf[col].mean() > 1000:
                                scaler = MinMaxScaler(feature_range=(0, 10))
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 100:
                                scaler = MinMaxScaler(feature_range=(0, 5))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            elif tmpdf[col].mean() > 10:
                                scaler = MinMaxScaler(feature_range=(0, 2))
                                # print(col)
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            else:
                                scaler = MinMaxScaler(feature_range=(0, 1))
                                tmpdf[col] = scaler.fit_transform(
                                    np.asarray(tmpdf[col]).reshape(-1, 1))
                            print("INFO : " + str(datetime.now()) +
                                  ' : ' + 'Column ' + col + 'is standardized')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return standardize_stratified_lvl

    def featurization(self, cat_coltype=False):
        def featurization_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.refresh_cat_noncat_cols_fn(tmpdf, self.threshold)
                    if cat_coltype:
                        column_list = self.catcols_list
                    else:
                        column_list = self.noncatcols_list
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before featurization ' + str(
                        tmpdf.shape))
                    for col in column_list:
                        tmpdf[col + '_minus_mean'] = tmpdf[col] - \
                            np.mean(tmpdf[col])
                        tmpdf[col + '_minus_mean'] = tmpdf[col +
                                                           '_minus_mean'].astype(np.float32)
                        tmpdf[col + '_minus_max'] = tmpdf[col] - \
                            np.max(tmpdf[col])
                        tmpdf[col + '_minus_max'] = tmpdf[col +
                                                          '_minus_max'].astype(np.float32)
                        tmpdf[col + '_minus_min'] = tmpdf[col] - \
                            np.min(tmpdf[col])
                        tmpdf[col + '_minus_min'] = tmpdf[col +
                                                          '_minus_min'].astype(np.float32)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after featurization ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return featurization_lvl1


    def feature_importance(self, dfforimp, tobepredicted, modelname, featurelimit=0):
        colname = [col for col in dfforimp.columns.tolist() if col !=
                   tobepredicted]
        X = dfforimp[colname]
        y = dfforimp[tobepredicted]
        # print(modelname)
        #t =''
        if modelname == 'rfclassifier':
            model = RandomForestClassifier(n_estimators=100, random_state=10)
        elif modelname == 'rfregressor':
            model = RandomForestRegressor(n_estimators=100, random_state=10)
        elif modelname == 'lgbmclassifier':
            model = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, verbose=-1)
        elif modelname == 'lgbmregressor':
            # print('yes')
            model = lgb.LGBMRegressor(
                n_estimators=1000, learning_rate=0.05, verbose=-1)
        else:
            print("Please specify the modelname")
        model.fit(X, y)
        feature_names = X.columns
        feature_importances = pd.DataFrame(
            {'feature': feature_names, 'importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(
            by=['importance'], ascending=False).reset_index()
        feature_importances = feature_importances[['feature', 'importance']]
        if featurelimit == 0:
            return feature_importances
        else:
            return feature_importances[:featurelimit]

    def importantfeatures(self, dfforimp, tobepredicted, modelname, skipcols=[], featurelimit=0):
        # print(modelname)
        f_imp = self.feature_importance(
            dfforimp, tobepredicted, modelname, featurelimit)
        allimpcols = list(f_imp['feature'])
        stuff = []
        for col in allimpcols:
            for skipcol in skipcols:
                if col != skipcol:
                    stuff.append(col)
                else:
                    pass
        return stuff, f_imp

    def convertdatatypes(self, cat_threshold=100):
        def convertdatatypes_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.dfcolumns_nottarget = [col for col in tmpdf.columns.tolist() if col != self.target]
                    for c in self.dfcolumns_nottarget:
                        col_dtype = tmpdf[c].dtype 
                        if (col_dtype == 'object') and (tmpdf[c].nunique() < cat_threshold):
                            tmpdf[c] = tmpdf[c].astype('category')
                        elif (col_dtype in ['int64','int32']) and (tmpdf[c].nunique() < cat_threshold):
                            tmpdf[c] = tmpdf[c].astype('category')
                        elif col_dtype in ['float64']:
                            tmpdf[c] = tmpdf[c].astype(np.float32)
                        elif col_dtype in ['int64',]:
                            tmpdf[c] = tmpdf[c].astype(np.int32)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return convertdatatypes_lvl

    def ohe_on_column(self, columns=None,drop_converted_col=True,refresh_cols=True):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    if columns is None:
                        if refresh_cols:
                            self.refresh_cat_noncat_cols_fn(tmpdf,self.threshold)
                        column_list = self.catcols_list
                    else:
                        column_list = columns
                    for col in column_list:
                        dummy = pd.get_dummies(tmpdf[col])
                        dummy.columns = [
                            col.lower() + '_' + str(x).lower().strip() +'_dums' for x in dummy.columns]
                        tmpdf = pd.concat([tmpdf, dummy], axis=1)
                        if drop_converted_col:
                            tmpdf = tmpdf.drop(col, axis=1)
                        print("INFO : " + str(datetime.now()) + ' : ' +
                              'Column ' + col + ' converted to dummies')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl   

    def remove_collinear(self, th=0.95):
        def converttodummies_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe before collinear drop ' + str(
                        tmpdf.shape))
                    corr_matrix = tmpdf.corr().abs()
                    upper = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                    to_drop = [column for column in upper.columns if any(
                        upper[column] > th)]
                    tmpdf = tmpdf.drop(to_drop, axis=1)
                    print("INFO : " + str(datetime.now()) + ' : ' + 'Shape of dataframe after collinear drop ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return converttodummies_lvl

    def high_coor_target_column(self, targetcol='y', th=0.5):
        def high_coor_target_column_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe before retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    cols = [col for col in tmpdf.columns.tolist() if col !=
                            targetcol]
                    for col in cols:
                        tmpdcorr = tmpdf[col].corr(tmpdf[targetcol])
                        if tmpdcorr > th:
                            self.hightarge_corr_col.append(col)
                    cols = self.hightarge_corr_col + [targetcol]
                    tmpdf = tmpdf[cols]
                    print("INFO : " + str(
                        datetime.now()) + ' : ' + 'Shape of dataframe after retaining only highly corelated col with target ' + str(
                        tmpdf.shape))
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return high_coor_target_column_lvl1

    def apply_agg_diff(self,aggfunc = 'median',quantile_val=0.5,columns=[]):
        def apply_agg_diff_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in columns:
                        if aggfunc == 'median':
                            diff_val = np.median(tmpdf[col])
                            tmpdf[f'{col}_mediandiff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        elif aggfunc == 'mean':
                            diff_val = np.mean(tmpdf[col])
                            tmpdf[f'{col}_meandiff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        elif aggfunc == 'min':
                            diff_val = np.min(tmpdf[col])
                            tmpdf[f'{col}_mindiff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        elif aggfunc == 'max':
                            diff_val = np.max(tmpdf[col])
                            tmpdf[f'{col}_maxdiff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        elif aggfunc == 'max':
                            diff_val = np.max(tmpdf[col])
                            tmpdf[f'{col}_maxdiff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        else:
                            diff_val = np.quantile(tmpdf[col],quantile_val)
                            tmpdf[f'{col}_q{quantile_val}diff'] = tmpdf[col].apply(lambda x: x - diff_val)
                        print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + f' converted using {aggfunc}')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) + ' : ' + sys.exc_info()[1])

            return wrapper

        return apply_agg_diff_lvl
    
    def logtransform(self, logtransform_col=[]):
        def logtransform_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in logtransform_col:
                        tmpdf[col] = tmpdf[col].apply(
                            lambda x: np.log(x) if x != 0 else 0)
                        print("INFO : " + str(
                            datetime.now()) + ' : ' + 'Column ' + col + ' converted to corresponding log using formula: log(x)')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper

        return logtransform_lvl
    
    def binary_target_encode(self, encode_save_path,encoding_cols=[],load_previous = False):
        def target_encode_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in encoding_cols:
                        if load_previous:
                            df = pd.read_csv(f'{encode_save_path}{col}.csv')
                        else:
                            
                            df = tmpdf[[col,self.target]].groupby([col]).sum().reset_index().sort_values(by=self.target,ascending=False)
                            df.columns = [col,f'{col}_tgt_enc']
                            df.to_csv(f'{encode_save_path}{col}.csv',index=False)
                        tmpdf = pd.merge(tmpdf,df,on=col,how='left')
                        tmpdf[col] = tmpdf[col].astype(np.float)
                        tmpdf[col] = tmpdf[col].fillna(0)
                        print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + ' target encoded')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper
        return target_encode_lvl

    def binary_target_ratio_encode(self, encode_save_path,encoding_cols=[],load_previous = False):
        def target_ratio_encode_lvl(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    for col in encoding_cols:
                        if load_previous:
                            df = pd.read_csv(f'{encode_save_path}{col}_tgt_ratio_enc.csv')
                        else:       
                            x = tmpdf[[col,self.target]].groupby([col]).sum().reset_index()
                            y = tmpdf[[col,self.target]].groupby([col]).count().reset_index()
                            df = pd.merge(x,y,on=col)
                            df[f'{col}_tgt_ratio_enc'] = df[f'{self.target}_x']/df[f'{self.target}_y']
                            df = df[[col,f'{col}_tgt_ratio_enc']]
                            df.to_csv(f'{encode_save_path}{col}_tgt_ratio_enc.csv',index=False)
                        tmpdf = pd.merge(tmpdf,df,on=col,how='left')
                        tmpdf[col] = tmpdf[col].astype(np.float)
                        tmpdf[col] = tmpdf[col].fillna(0)
                        print("INFO : " + str(datetime.now()) + ' : ' + 'Column ' + col + ' target encoded')
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])
            return wrapper
        return target_ratio_encode_lvl

    def refresh_cat_noncat_cols_fn(self, tmpdf, cat_threshold=100):
        try:
            self.catcols = {}
            self.catcols_list = []
            self.noncatcols = {}
            self.noncatcols_list = []
            self.dfcolumns = tmpdf.columns.tolist()
            self.dfcolumns_nottarget = [
                col for col in self.dfcolumns if col != self.target]
            for col in self.dfcolumns_nottarget:
                col_unique_cnt = tmpdf[col].nunique()
                if (col_unique_cnt < cat_threshold) and (
                        (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                    self.catcols.update({col: col_unique_cnt})
                    self.catcols_list.append(col)
                else:
                    self.noncatcols.update({col: col_unique_cnt})
                    self.noncatcols_list.append(col)
        except:
            sys.exit('ERROR : ' + str(datetime.now()) +
                     ' : ' + sys.exc_info()[1])

    def refresh_cat_noncat_cols(self, cat_threshold):
        def refresh_cat_noncat_cols_lvl1(func):
            def wrapper(tmpdf):
                try:
                    tmpdf = func(tmpdf)
                    self.catcols = {}
                    self.catcols_list = []
                    self.noncatcols = {}
                    self.noncatcols_list = []
                    self.dfcolumns = tmpdf.columns.tolist()
                    self.dfcolumns_nottarget = [
                        col for col in self.dfcolumns if col != self.target]
                    for col in self.dfcolumns_nottarget:
                        col_unique_cnt = tmpdf[col].nunique()
                        if (col_unique_cnt < cat_threshold) and (
                                (tmpdf[col].dtype != 'float32') and (tmpdf[col].dtype != 'float64')):
                            self.catcols.update({col: col_unique_cnt})
                            self.catcols_list.append(col)
                        else:
                            self.noncatcols.update({col: col_unique_cnt})
                            self.noncatcols_list.append(col)
                    return tmpdf
                except:
                    sys.exit('ERROR : ' + str(datetime.now()) +
                             ' : ' + sys.exc_info()[1])

            return wrapper
        return refresh_cat_noncat_cols_lvl1
