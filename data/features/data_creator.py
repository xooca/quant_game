from functools import partial
from datetime import datetime
import zipfile,fnmatch,os
import pandas as pd
import pickle
from pathlib import Path
from data.features import data_utils as du
from config.common.config import Config,DefineConfig


def convert_date(indate,datefmt="%Y%m%d %H:%M"):
    return datetime.strptime(indate, datefmt)
    
def create_timeframe_df(data,time_frames,date_fmt,date_col,time_col,base_location,saved_file_prefix='data',save_data = True, return_dfs = False):
    paths =[]
    ret_dfs = []
    for time_frame in time_frames:
        if time_frame == '3min':
            tmpdf = data[data[time_col].str[-2:].isin(['03','06','09','12','15','18','21','24','27','30','33','36','39','42','45','48','51','54','57','60'])].reset_index(drop=True)
        elif time_frame == '5min':
            tmpdf = data[data[time_col].str[-1:].isin(['0','5'])].reset_index(drop=True)
        elif time_frame == '10min':
            tmpdf = data[data[time_col].str[-1:].isin(['0','5'])].reset_index(drop=True)
        elif time_frame == '15min':
            tmpdf = data[data[time_col].str[-2:].isin(['15','30','45','00'])].reset_index(drop=True)
        elif time_frame == '30min':
            tmpdf = data[data[time_col].str[-2:].isin(['30','00'])].reset_index(drop=True)
        elif time_frame == '1hour':
            tmpdf = data[data[time_col].str[-2:].isin(['00'])].reset_index(drop=True)
        elif time_frame  == 'openinghr':
            tmpdf = data[data[time_col].str[:2].isin(['09'])].reset_index(drop=True)
        elif time_frame  == 'lasthr':
            tmpdf = data[data[time_col].str[:2].isin(['15'])].reset_index(drop=True)
        
        f = partial(convert_date,datefmt = date_fmt)
        tmpdf[date_col] = tmpdf[date_col].astype('str') + ' '+ tmpdf[time_col].astype('str')
        tmpdf[date_col] = tmpdf[date_col].apply(f)
        tmpdf.index = tmpdf[date_col]
        tmpdf = tmpdf.drop([date_col,time_col],axis=1)
        if save_data:
            filepath = f'{base_location}{saved_file_prefix}_{time_frame}.csv'
            tmpdf.to_csv(filepath)
            print(f"Save at location {filepath}")
        paths.append(filepath)
        if return_dfs:
            ret_dfs.append(tmpdf)
    return paths, ret_dfs

class Initial_Dataset():
    def __init__(self,conn,zip_file_path,zip_file_pattern,ticker_list):
        self.zip_file_path = zip_file_path
        self.zip_file_pattern = zip_file_pattern
        self.ticker_list = ticker_list
        self.conn = conn

    def unzip_folders(self):
        self.zipfiles_loaded_list = []
        self.zipfiles_notloaded_list = []
        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.zip_file_pattern):
                f_name = os.path.join(root, filename)
                try:
                    if zipfile.is_zipfile(f_name):
                        n_file = os.path.join(root, os.path.splitext(filename)[0])
                        zipfile.ZipFile(f_name).extractall(n_file)
                        du.print_log(f"File saved at location {n_file}",self.using_print)
                        self.zipfiles_loaded_list.append(f_name)
                        os.remove(f_name)
                        du.print_log(f"File {f_name} removed",self.using_print)
                    else:
                        du.print_log(f"File {f_name} is not unzipped",self.using_print)
                        self.zipfiles_notloaded_list.append(f_name)
                except Exception as e1:
                    du.print_log(f"File {f_name} is not unzipped",self.using_print)
                    self.zipfiles_notloaded_list.append(f_name)
                    du.print_log(f"Error encountered is {e1}",self.using_print)
        zip_files_df = pd.DataFrame()
        notloaded_df = pd.DataFrame({'files':zipfiles_notloaded_list,'status':['notloaded']*len(zipfiles_notloaded_list)})
        loaded_df = pd.DataFrame({'files':zipfiles_loaded_list,'status':['loaded']*len(zipfiles_loaded_list)})
        files_df = pd.concat([notloaded_df,loaded_df])
        du.write_df_to_table(connection=self.conn,table_name='metadata.zipfile',files_df)

    def create_dataset(self,reload_all = True):
        files_list = []
        bad_files = []
        files_processed = []
        base_df = pd.DataFrame(columns = self.initial_columns)
        du.print_log(f'Source data path is { self.source_data}')
        if not os.path.exists(self.source_data):
            os.makedirs(self.source_data)
            du.print_log(f'Created folder {self.source_data}',self.using_print)

        already_loaded_file_name = f'{self.source_data}already_loaded_files.pickle'
        du.print_log(f'Data save path is { self.raw_data_save_path}')
        du.print_log(f'File with already loaded files is {already_loaded_file_name}')
        try:
            with open(already_loaded_file_name, 'rb') as handle:
                already_loaded_files = pickle.load(handle)
                already_loaded_files = [Path(col) for col in already_loaded_files]
                du.print_log(f"Total files already saved {len(already_loaded_files)}",self.using_print)
        except Exception as e1:
            du.print_log(f"File {already_loaded_file_name} is not loaded because of error : {e1}",self.using_print)
            already_loaded_files = []
        du.print_log(f"Raw data root path is {self.raw_data_input_path}",self.using_print)
        for root, dirs, files in os.walk(self.raw_data_input_path):
            for filename in fnmatch.filter(files, self.data_pattern):
                f_name = Path(os.path.join(root, filename))
                files_list.append(f_name)

        files_to_be_loaded = [f for f in files_list if f not in already_loaded_files]
        files_to_be_loaded = list(dict.fromkeys(files_to_be_loaded))
        files_list = list(dict.fromkeys(files_list))
        du.print_log(f"Total files detected {len(files_list)}",self.using_print)
        du.print_log(f"Total new files detected {len(files_to_be_loaded)}",self.using_print)
        
        try:
            base_df = pd.read_csv(self.raw_data_save_path)
        except Exception as e1:
            du.print_log(f"Error while loading dataframe from { self.raw_data_save_path} because of error : {e1}")
            base_df = pd.DataFrame(columns = self.ohlc_column)
            files_to_be_loaded = files_list
        if len(base_df) == 0 or reload_all:
            files_to_be_loaded = files_list
            du.print_log(f"We are going to reload all the data",self.using_print)
        du.print_log(f"Number of files to be loaded {len(files_to_be_loaded)}",self.using_print)
        base_df_st_shape = base_df.shape
        files_to_be_loaded = sorted(files_to_be_loaded)
        for i,f_name in enumerate(files_to_be_loaded,1):
            f_name = os.path.join(root, f_name)
            try:
                tmp_df = pd.read_csv(f_name,header=None)
                tmp_df = tmp_df.loc[:,0:6]
                tmp_df.columns = self.initial_columns
                tmp_df = du.convert_df_to_timeseries(tmp_df)
                base_df = pd.concat([base_df,tmp_df],axis=0)
                du.print_log(f"Data shape after loading file {f_name} is {base_df.shape}",self.using_print)
                du.print_log(f"Files left to be loaded {len(files_to_be_loaded)-i}",self.using_print)
                already_loaded_files.append(f_name)
            except Exception as e1:
                bad_files.append(f_name)
                du.print_log(f"File {f_name} is not loaded because of error : {e1}",self.using_print)
        with open(already_loaded_file_name, 'wb') as handle:
            pickle.dump(already_loaded_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        du.print_log(f"Shape of the dataframe before duplicate drop is {base_df.shape}",self.using_print)
        base_df = base_df.drop_duplicates()
        du.print_log(f"Shape of the dataframe after duplicate drop is {base_df.shape}",self.using_print)
        #if base_df_st_shape != base_df.shape:
        base_df = base_df.sort_index()
        base_df.to_csv( self.raw_data_save_path)
        du.print_log(f"Saving dataframe to location { self.raw_data_save_path}",self.using_print)
        return base_df
