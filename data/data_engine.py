from re import A
from statistics import mean
from tabnanny import verbose
#from signal import Signal
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime as dt
from talib.abstract import *
import zipfile,fnmatch,os
import pandas as pd
import pickle
from pathlib import Path
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer,CategoricalImputer,ArbitraryNumberImputer,EndTailImputer,DropMissingData
from data.signals import Signals,add_all_ta_features
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def print_log(log,using_print=True):
    if using_print:
        print(log)
    else:
        logging.info(log)
        
def unzip_folders(rootPath,pattern):
    for root, dirs, files in os.walk(rootPath):
        for filename in fnmatch.filter(files, pattern):
            print_log(os.path.join(root, filename))
            zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))
            os.remove(os.path.join(root, filename))

def convert_df_to_timeseries(df):
    df['date_time'] = df['date'].astype(str) + ' ' + df['time']
    df = df.sort_values(by='date_time')
    df.index = df['date_time']
    df = df[['open','high','low','close']]
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    return df

def create_dataset(root_path,pattern,data_save_path,data_name,reset_df = False):
    files_list = []
    bad_files = []
    files_processed = []
    base_df = pd.DataFrame(columns = ['name','date','time','open','high','low','close'])
    if not os.path.exists(f'{data_save_path}{data_name}/'):
        os.makedirs(f'{data_save_path}{data_name}/')
        print_log(f'Created folder {data_save_path}{data_name}')

    already_loaded_file_name = f'{data_save_path}{data_name}/already_loaded_files.pickle'
    csv_save_location = f'{data_save_path}{data_name}/{data_name}.csv'
    print_log(f'Data save path is {csv_save_location}')
    print_log(f'File with already loaded files is {already_loaded_file_name}')
    orig_cols = ['name','date','time','open','high','low','close']
    try:
        with open(already_loaded_file_name, 'rb') as handle:
            already_loaded_files = pickle.load(handle)
            already_loaded_files = [Path(col) for col in already_loaded_files]
            print_log(f"Total files already saved {len(already_loaded_files)}")
    except Exception as e1:
        print_log(f"File {already_loaded_file_name} is not loaded because of error : {e1}")
        already_loaded_files = []
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            f_name = Path(os.path.join(root, filename))
            files_list.append(f_name)
    files_to_be_loaded = [f for f in files_list if f not in already_loaded_files]
    files_to_be_loaded = list(dict.fromkeys(files_to_be_loaded))
    files_list = list(dict.fromkeys(files_list))
    print_log(f"Total files detected {len(files_list)}")
    print_log(f"Total new files detected {len(files_to_be_loaded)}")
    try:
        base_df = pd.read_csv(csv_save_location)
    except Exception as e1:
        print_log(f"Error while loading dataframe from {csv_save_location} because of error : {e1}")
        base_df = pd.DataFrame(columns = ['open','high','low','close'])
        files_to_be_loaded = files_list
    if len(base_df) == 0 or reset_df:
        files_to_be_loaded = files_list
        print_log(f"We are going to reload all the data")

    print_log(f"Number of files to be loaded {len(files_to_be_loaded)}")
    base_df_st_shape = base_df.shape
    for i,f_name in enumerate(files_to_be_loaded,1):
        f_name = os.path.join(root, f_name)
        try:
            tmp_df = pd.read_csv(f_name,header=None)
            tmp_df = tmp_df.loc[:,0:6]
            tmp_df.columns = orig_cols
            tmp_df = convert_df_to_timeseries(tmp_df)
            base_df = pd.concat([base_df,tmp_df],axis=0)
            print_log(len(files_to_be_loaded)-i,base_df.shape,f_name)
            already_loaded_files.append(f_name)
        except Exception as e1:
            bad_files.append(f_name)
            print_log(f"File {f_name} is not loaded because of error : {e1}")
    with open(already_loaded_file_name, 'wb') as handle:
        pickle.dump(already_loaded_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_log(f"Shape of the dataframe before duplicate drop is {base_df.shape}")
    base_df = base_df.drop_duplicates()
    print_log(f"Shape of the dataframe after duplicate drop is {base_df.shape}")
    if base_df_st_shape != base_df.shape:
        base_df = base_df.sort_index()
        base_df.to_csv(csv_save_location, index_label=False )
        print_log(f"Saving dataframe to location {csv_save_location}")
    return base_df

class LabelCreator(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def label_generator_v2(self,val):
        if val <= 10 and val>=-10:
            return '-10to10'
        elif val > 10 and val <= 20:
            return '10to20'
        elif val > 20 and val <= 40:
            return '20to40'
        elif val > 40 and val <= 60:
            return '40to60'
        elif val > 60 and val <= 80:
            return '60to80'
        elif val > 80 and val <= 100:
            return '80to100'
        elif val > 100:
            return 'above100'
        elif val < -10 and val >= -20:
            return '-10to-20'
        elif val < -20 and val >= -40:
            return '-20to-40'
        elif val < -40 and val >= -60:
            return '-40to-60'
        elif val < -60 and val >= -80:
            return '-60to-80'
        elif val < -80 and val >= -100:
            return '-80to-100'
        elif val < -100:
            return 'below100'
        else:
            return 'unknown'

    def label_generator(self,val):
        if val <= 35 and val>=0:
            return '-0to35'
        elif val > 35 and val <= 80:
            return '35to80'
        elif val > 80 and val <= 150:
            return '80to150'
        elif val > 150:
            return 'above150'
        elif val > -35 and val <= 0:
            return '0to-35'
        elif val > -80 and val <= -35:
            return '-35to-80'
        elif val > -150 and val <= -80:
            return '-80to-150'
        elif val < -150:
            return 'below150'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df

class LabelCreator_Light(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def label_generator(self,val):
        if val <= 30 and val>=0:
            return '-0to30'
        elif val > 30 and val <= 80:
            return '30to80'
        elif val > 80:
            return 'above80'
        elif val > -30 and val <= 0:
            return '0to-30'
        elif val > -80 and val <= -30:
            return '-30to-80'
        elif val <= -80:
            return 'below80'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df

class LabelCreator_Super_Light(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}_SL'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def label_generator(self,val):
        if val <= 30 and val>=-30:
            return 'neutral'
        elif val > 30:
            return 'call'
        elif val < -30:
            return 'put'
        else:
            return 'unknown'

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print_log(f"Shape of dataframe after transform is {df.shape}") 
        return df
class TechnicalIndicator(BaseEstimator, TransformerMixin):
    def __init__(self,method_type = ['volumn_','volatile_','transform_','cycle_','pattern_','stats_','math_','overlap_']):
        self.method_type = method_type

    def fit(self, X, y=None):
        self.all_methods = []
        a = dict(Signals.__dict__)
        for a1,a2 in a.items():
            self.all_methods.append(a1)
        self.all_methods = [m1 for m1,m2 in a.items() if m1[:1]!='_']
        self.all_methods = [m for m in self.all_methods for mt in self.method_type if mt in m]
        return self    # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        sig = Signals(df)
        self.methods_run = []
        self.methods_notrun = []
        for f in self.all_methods:
            try:
                exec(f'sig.{f}()')
                self.methods_run.append(f)
            except Exception as e1:
                print_log(f"Function {f} was unable to run, Error is {e1}")
                self.methods_notrun.append(f)
        print_log(f"Shape of dataframe after TechnicalIndicator is {df.shape}")
        return sig.df

class CreateTechnicalIndicatorUsingPandasTA(BaseEstimator, TransformerMixin):
    def __init__(self,exclude=["pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]
,verbose=True):
        import pandas_ta as ta
        self.exclude = exclude
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        import pandas_ta as ta
        if self.verbose:
            print_log(f"Shape of dataframe before CreateTechnicalIndicatorUsingPandasTA is {df.shape}")
        df.ta.strategy(exclude=self.exclude,verbose=self.verbose,timed=True)
        if self.verbose:
            print_log(f"Shape of dataframe after CreateTechnicalIndicatorUsingPandasTA is {df.shape}") 
        return df

class CreateTechnicalIndicatorUsingTA(BaseEstimator, TransformerMixin):
    def __init__(self, open='open',high='high',low='low',close='close',volume='volume',vectorized=True,fillna=False,colprefix='ta',volume_ta=True,volatility_ta=True,trend_ta=True,momentum_ta=True,others_ta=True,verbose=True):
        self.open=open
        self.high=high
        self.low=low
        self.close=close
        self.volume=volume
        self.fillna=fillna
        self.colprefix=colprefix
        self.volume_ta=volume_ta
        self.volatility_ta=volatility_ta
        self.trend_ta=trend_ta
        self.momentum_ta=momentum_ta
        self.others_ta=others_ta
        self.verbose = verbose
        self.vectorized = vectorized
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        if self.verbose:
            print_log(f"Shape of dataframe before CreateTechnicalIndicatorUsingTA is {df.shape}")
        df = add_all_ta_features(
            df,
            open = self.open,
            high = self.high,
            low = self.low,
            close = self.close,
            volume = self.volume,
            fillna = self.fillna,
            colprefix = self.colprefix,
            vectorized = self.vectorized,
            volume_ta = self.volume_ta,
            volatility_ta  = self.volatility_ta,
            trend_ta  = self.trend_ta,
            momentum_ta  = self.momentum_ta,
            others_ta = self.others_ta,
        )
        if self.verbose:
            print_log(f"Shape of dataframe after CreateTechnicalIndicatorUsingTA is {df.shape}") 
        return df
class NormalizeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern = [],columns = [],impute_values=False,impute_type = 'categorical',convert_to_floats = False,arbitrary_impute_variable=99,drop_na_col=False,drop_na_rows=False,
    fillna = False,fillna_method = 'bfill',fill_index=False):
        self.impute_values = impute_values
        self.convert_to_floats = convert_to_floats
        self.impute_type = impute_type
        self.arbitrary_impute_variable = arbitrary_impute_variable
        self.drop_na_col = drop_na_col
        self.drop_na_rows = drop_na_rows
        self.fillna_method = fillna_method
        self.fillna = fillna
        self.column_pattern = column_pattern
        self.columns = columns
        self.fill_index = fill_index


    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        print_log(f"Shape of dataframe before NormalizeDataset is {df.shape}")
        info_list = []
        df = convert_todate_deduplicate(df)
        if self.convert_to_floats:
            for col in self.columns:
                df[col] = df[col].astype('float')
                info_list.append('convert_to_floats')
        if self.fill_index:
            df = df.reindex(pd.date_range(min(df.index), max(df.index), freq ='1min'))
            df = df.resample('1min').ffill()
        if self.impute_values:
            from sklearn.pipeline import Pipeline
            if self.impute_type == 'mean_median_imputer':
                imputer = MeanMedianImputer(imputation_method='median', variables=self.columns)
                info_list.append('mean_median_imputer')
            elif self.impute_type == 'categorical':
                imputer = CategoricalImputer(variables=self.columns)
                info_list.append('categorical')
            elif self.impute_type == 'arbitrary':
                if isinstance(self.arbitrary_impute_variable, dict):
                    imputer = ArbitraryNumberImputer(imputer_dict = self.arbitrary_impute_variable)
                    
                else:
                    imputer = ArbitraryNumberImputer(variables = self.columns,arbitrary_number = self.arbitrary_number)
                info_list.append('arbitrary')
            else:
                imputer = CategoricalImputer(variables=self.columns)
                info_list.append('categorical')
            imputer.fit(df)
            df= imputer.transform(df)
        if self.fillna:
            df = df.fillna(method=self.fillna_method)
            info_list.append('fillna')
        if self.drop_na_col:
            imputer = DropMissingData(missing_only=True)
            imputer.fit(df)
            df= imputer.transform(df)
            info_list.append('drop_na_col')
        if self.drop_na_rows:
            #df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            df = df.dropna(axis=0)
            info_list.append('drop_na_rows')
        df = df.sort_index()
        print_log(f"Shape of dataframe after NormalizeDataset is {df.shape} : {'.'.join(info_list)}")
        return df
class LastTicksGreaterValuesCount(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],create_new_col = True,last_ticks=10):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self   # Nothing to do in fit in this scenario
    
    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def transform(self, df):
        print_log('*'*100)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]         
        for col in self.columns:
            print_log(f"LastTicksGreaterValuesCount : {col} : f'last_tick_{col}_{self.last_ticks}'")
            x = np.concatenate([[np.nan] * (self.last_ticks), df[col].values])
            arr = self.rolling_window(x, self.last_ticks + 1)
            #print_log(arr)
            if self.create_new_col:
                #df[f'last_tick_{col}_{self.last_ticks}'] = self.rolling_window(x, self.#last_ticks + 1)
                df[f'last_tick_{col}_{self.last_ticks}']  = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
            else:
                df[col] = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
        print_log(f"Shape of dataframe after LastTicksGreaterValuesCount is {df.shape}")
        return df

def convert_todate_deduplicate(df):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')] 
    return df

class PriceLastTickBreachCountv1(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],create_new_col = True,last_ticks='10min',breach_type = ['mean']):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.breach_type = breach_type
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        #df.index = pd.to_datetime(df.index)
        #df = df.sort_index()
        #df = df[~df.index.duplicated(keep='first')]         
        for col in self.columns:
            #print_log(f"PriceLastTickBreachCount : {col}")
            for breach_type in self.breach_type:
                
                if self.create_new_col:
                    col_name = f'last_tick_{breach_type}_{col}_{self.last_ticks}'
                else:
                    col_name = col
                print_log(f"PriceLastTickBreachCount : {breach_type} : {col} : {col_name}")
                if breach_type == 'morethan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
                elif breach_type == 'lessthan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
                elif breach_type == 'mean':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
                elif breach_type == 'min':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
                elif breach_type == 'max':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
                elif breach_type == 'median':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
                elif breach_type == '10thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
                elif breach_type == '25thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
                elif breach_type == '75thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
                elif breach_type == '95thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
                else:
                    df[col_name] = (df[col].rolling(self.last_ticks, min_periods=1)
                            .apply(lambda x: (x[-1] > x[:-1]).mean())
                            .astype(int))
        print_log(f"Shape of dataframe after PriceLastTickBreachCount is {df.shape}")
        return df

class ValueLastTickBreachCount(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[],column_pattern=[],create_new_col = True,last_ticks='5',breach_type = ['morethan'],verbose=False):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.breach_type = breach_type
        self.self.verbose = self.verbose
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):   
        print_log('*'*100)
        df = df.sort_index()    
        for col in self.columns:
            for breach_type in self.breach_type:
                
                if self.create_new_col:
                    col_name = f'last_tick_{breach_type}_{col}_{self.last_ticks}'
                else:
                    col_name = col
                print_log(f"ValueLastTickBreachCount : {breach_type} : {col} : {col_name}")
                if breach_type == 'morethan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
                elif breach_type == 'lessthan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
                elif breach_type == 'mean':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
                elif breach_type == 'min':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
                elif breach_type == 'max':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
                elif breach_type == 'median':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
                elif breach_type == '10thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
                elif breach_type == '25thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
                elif breach_type == '75thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
                elif breach_type == '95thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
                else:
                    df[col_name] = (df[col].rolling(self.last_ticks, min_periods=1)
                            .apply(lambda x: (x[-1] > x[:-1]).mean())
                            .astype(int))
        if self.self.verbose:
            print_log(f"Shape of dataframe after ValueLastTickBreachCount is {df.shape}")
        return df

class PriceLastTickBreachCount(BaseEstimator, TransformerMixin):
    def __init__(self, column_pattern=[],columns=[],last_ticks='10min',breach_type = ['mean']):
        self.columns = columns
        self.last_ticks = last_ticks
        self.breach_type = breach_type
        self.column_pattern = column_pattern
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()   
        for breach_type in self.breach_type:
            print_log(f"PriceLastTickBreachCount : {breach_type} : {self.last_ticks}")
            if breach_type == 'morethan':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] > np.array(x[:-1]))).fillna(0)
            elif breach_type == 'lessthan':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] < np.array(x[:-1]))).fillna(0)
            elif breach_type == 'mean':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.mean(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'min':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.min(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'max':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.max(np.array(x)))).fillna(0).astype(int)
            elif breach_type == 'median':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.median(np.array(x)))).fillna(0).astype(int)
            elif breach_type == '10thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.1))).fillna(0).astype(int)
            elif breach_type == '25thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.25))).fillna(0).astype(int)
            elif breach_type == '75thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.75))).fillna(0).astype(int)
            elif breach_type == '95thquantile':
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x > np.quantile(np.array(x),0.95))).fillna(0).astype(int)
            else:
                tmpdf = df[self.columns].rolling(self.last_ticks, min_periods=1).apply(lambda x: sum(x[-1] > np.array(x[:-1]))).fillna(0)
            col_names = [f"{col}_{self.last_ticks}_{'_'.join(self.breach_type)}_last_tick_" for col in self.columns]
            tmpdf.columns = col_names
            df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        print_log(f"Shape of dataframe after PriceLastTickBreachCount is {df.shape}")
        return df

class RollingValues(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[],column_pattern=[],last_ticks=['5min','10min'],aggs=['mean','max'],oper = ['-','='],verbose=True):
        self.columns = columns
        self.last_ticks = last_ticks
        self.verbose = verbose
        self.column_pattern = column_pattern
        self.aggs = aggs
        self.oper = oper
        
    def fit(self, df, y=None):
        if len(self.columns) == 0:
            self.columns = [m for m in df.columns.tolist() for mt in self.column_pattern if mt in m]
            self.columns = list(set(self.columns))
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):   
        print_log('*'*100)
        df = df.sort_index()
        eval_stmt = '' 
        for lt,oper,agg in zip(self.last_ticks,self.oper,self.aggs):
            #print_log(lt,oper,agg)
            tmpst = f"df[{self.columns}].rolling('{lt}', min_periods=1).{agg}() {oper}"
            eval_stmt = eval_stmt + tmpst
        tmpdf = eval(eval_stmt[:-1])
        col_names = [f"{shftcol}_{'_'.join(self.last_ticks)}_{'_'.join(self.aggs)}_rolling_values" for shftcol in self.columns]
        tmpdf.columns = col_names
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after RollingValues is {df.shape}")
        return df

class PriceDayRangeHourWise(BaseEstimator, TransformerMixin):
    def __init__(self, first_col = 'high',second_col='low',hour_range = [['09:00', '10:30'],['10:30', '11:30']],range_type=['price_range','price_deviation_max_first_col']):
        self.hour_range = hour_range
        self.first_col = first_col
        self.second_col = second_col
        self.range_type = range_type
        
    def fit(self, X, y=None):
        return self    

    def transform(self, df):
        print_log('*'*100)
        #df = convert_todate_deduplicate(df)
        df = df.sort_index()
        for r1,r2 in self.hour_range:
            for rt in self.range_type:
                print_log(f"PriceDayRangeHourWise : {self.first_col} : {self.second_col} : {r1} : {r2} : {rt}")
                if rt == 'price_range':
                    #print_log(df[self.first_col])
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                else:
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
            s1.index = pd.to_datetime(s1.index) 
            s1 = s1.sort_index()
            c = [int(i) for i in r2.split(':')]
            s1.index = s1.index + pd.DateOffset(minutes=c[0]*60 + c[1])
            col_name = f"PDR_{self.first_col}_{self.second_col}_{rt}_{r1.replace(':','')}_{r2.replace(':','')}"
            s1.name = col_name
            df=pd.merge(df,s1, how='outer', left_index=True, right_index=True)
            df[col_name] = df[col_name].fillna(method='ffill')
        print_log(f"Shape of dataframe after PriceDayRangeHourWise is {df.shape}")
        return df

class PriceVelocityv2(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            
            if self.freq is not None:
                self.col_name = f'price_velocity_{shftcol}_{self.freq}_{self.shift}'
                print_log(f"PriceVelocity : {shftcol} : {self.col_name}")
                a = df.shift(self.shift, freq=self.freq)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name].round(3)
            else:
                self.col_name = f'price_velocity_{shftcol}_{self.shift}'
                print_log(f"PriceVelocity : {shftcol} : {self.col_name}")
                a = df.shift(self.shift)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name].round(3)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PriceVelocity(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.freq is not None:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift,freq=self.freq)[self.shift_column])
            col_names = [f'{shftcol}_{self.freq}_{self.shift}_price_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        else:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift)[self.shift_column])
            col_names = [f'{shftcol}_{self.shift}_price_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PriceVelocityv1(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            print_log(f"PriceVelocity : {shftcol}")
            if self.freq is not None:
                self.col_name = f'price_velocity_{shftcol}_{self.freq}_{self.shift}'
                #df[self.col_name] = df[shftcol].subtract(df.shift(self.shift, freq=self.freq)[shftcol])
                df[self.col_name] = df[shftcol] - df.shift(self.shift, freq=self.freq)[shftcol]
                df[self.col_name] = df[self.col_name].round(3)
            else:
                self.col_name = f'price_velocity_{shftcol}_{self.shift}'
                #df[self.col_name] = df[shftcol].subtract(df.shift(self.shift)[shftcol])
                df[self.col_name] = df[shftcol] - df.shift(self.shift)[shftcol]
                df[self.col_name] = df[self.col_name].round(3)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PricePerIncrementv1(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        for shftcol in self.shift_column:
            
            if self.freq is not None:
                self.col_name = f'price_pervelocity_{shftcol}_{self.freq}_{self.shift}'
                print_log(f"PricePerIncrement : {shftcol} : {self.col_name}")
                a = df.shift(self.shift, freq=self.freq)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name]/int(self.shift)
                df[self.col_name] = df[self.col_name].round(4)
            else:
                self.col_name = f'price_pervelocity_{shftcol}_{self.shift}'
                print_log(f"PricePerIncrement : {shftcol} : {self.col_name}")
                a = df.shift(self.shift)[shftcol]
                a.name = self.col_name
                df = pd.merge(df, a, left_index=True, right_index=True,how='left')
                df[self.col_name] = df[shftcol] - df[self.col_name]
                df[self.col_name] = df[self.col_name]/int(self.shift)
                df[self.col_name] = df[self.col_name].round(4)
        if self.verbose:
            print_log(f"Shape of dataframe after PriceVelocity is {df.shape}") 
        return df

class PricePerIncrement(BaseEstimator, TransformerMixin):
    def __init__(self, freq='D',shift=5,shift_column=['close','open'],shift_column_pattern=[],verbose=False):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.verbose = verbose
        self.shift_column_pattern = shift_column_pattern
        
    def fit(self, df, y=None):
        if len(self.shift_column) == 0:
            self.shift_column = [m for m in df.columns.tolist() for mt in self.shift_column_pattern if mt in m]
            self.shift_column = list(set(self.shift_column))
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.freq is not None:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift,freq=self.freq)[self.shift_column])
            
            col_names = [f'{shftcol}_{self.freq}_{self.shift}_price_per_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        else:
            tmpdf = df[self.shift_column].subtract(df.shift(self.shift)[self.shift_column])
            col_names = [f'{shftcol}_{self.shift}_price_per_velocity' for shftcol in self.shift_column]
            tmpdf.columns = col_names
        tmpdf = tmpdf/self.shift
        df = pd.merge(df, tmpdf, left_index=True, right_index=True,how='left')
        if self.verbose:
            print_log(f"Shape of dataframe after PricePerVelocity is {df.shape}") 
        return df

class FilterData(BaseEstimator, TransformerMixin):
    def __init__(self, start_date=None,end_date=None,filter_rows=None,verbose=False):
        self.start_date = start_date
        self.end_date = end_date
        self.filter_rows = filter_rows
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        print_log(f"Value of start_date is {self.start_date}") 
        print_log(f"Value of end_date is {self.end_date}") 
        print_log(f"Value of filter_rows is {self.filter_rows}")
        if self.verbose:
            print_log(f"Shape of dataframe before FilterData is {df.shape}") 
        if self.start_date != 'None' and self.end_date == 'None':
            df = df.sort_index().loc[self.start_date:]
            print_log(f"Data filtered with {self.start_date}") 
        elif self.start_date == 'None' and self.end_date != 'None':
            df = df.sort_index().loc[:self.end_date]
            print_log(f"Data filtered with {self.end_date}") 
        elif self.start_date != 'None' and self.end_date != 'None':
            df = df.sort_index().loc[self.start_date:self.end_date]
            print_log(f"Data filtered with {self.end_date}") 
        else:
            df = df.sort_index()
            print_log(f"No filtering done") 
        if self.filter_rows != 'None':
            df = df[:self.filter_rows]
            print_log(f"Data filtered with filter rows {self.filter_rows}") 
        if self.verbose:
            print_log(f"Shape of dataframe after FilterData is {df.shape}") 
        return df

class Zscoring(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window = 30,verbose=False):
        self.columns = columns
        self.verbose = verbose
        self.window = window
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def zscore(self,x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x-m)/s
        return z

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before Zscoring is {df.shape}") 
        #zscore_fxn = lambda x: (x - x.rolling(window=200, min_periods=20).mean())/ x.rolling(window=200, min_periods=20).std()
        #for col in self.columns:
        #    df[f'Zscore_{col}_{self.window}'] =self.zscore(df[col],self.window)
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'Zscore_{col}_{self.window}':self.zscore(df[col],self.window)})
            print_log(f"Zscore_{col}_{self.window} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Zscoring is {df.shape}") 
        return df

class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,verbose=False):
        self.columns = columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before LogTransform is {df.shape}") 
        for col in self.columns:
            df[f'Log_{col}'] =df[col].apply(np.log)
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'Log_{col}':df[col].apply(np.log)})
            print_log(f"Log_{col} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after LogTransform is {df.shape}") 
        return df

class PercentageChange(BaseEstimator, TransformerMixin):
    def __init__(self, columns,periods=30, fill_method='pad', limit=None, freq=None,verbose=False):
        self.columns = columns
        self.periods = periods
        self.fill_method = fill_method
        self.limit = limit
        self.freq = freq
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PercentageChange is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PerChg_{col}_{self.periods}_{self.freq}':df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)})
            #df[f'PerChg_{col}_{self.periods}_{self.freq}'] =df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)
            print_log(f"PerChg_{col}_{self.periods}_{self.freq} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PercentageChange is {df.shape}") 
        return df

class WeightedExponentialAverage(BaseEstimator, TransformerMixin):
    def __init__(self, columns,com=None, span=44, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None, verbose=False):
        self.columns = columns
        self.com = com
        self.span = span
        self.halflife = halflife
        self.alpha = alpha
        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.axis = axis
        self.times = times
        #self.method = method
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before WeightedExponentialAverage is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'WEA_{col}_{self.span}':df[col].ewm(com=self.com, span=self.span, halflife=self.halflife, alpha=self.alpha, min_periods=self.min_periods, adjust=self.adjust, ignore_na=self.ignore_na, axis=self.axis, times=self.times).mean()})
            print_log(f"WEA_{col}_{self.span} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after WeightedExponentialAverage is {df.shape}") 
        return df

class PercentileTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,quantile=0.75,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.quantile = quantile
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PercentileTransform is {df.shape}")
        #rollrank_fxn = lambda x: x.rolling(self.window, min_periods=self.min_periods).apply(lambda x: pd.Series(x).quantile(0.75))
        #for col in self.columns:
        #    df[f'PCTL_{col}_{self.window}_{self.min_periods}'] =df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: pd.Series(x).quantile(self.quantile))
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PCTL_{col}_{self.window}_{self.min_periods}':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1] - np.quantile(np.array(x),q=self.quantile))})
            print_log(f"PCTL_{col}_{self.window}_{self.min_periods} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PercentileTransform is {df.shape}") 
        return df

class RollingRank(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def rank(self,s):
        #s = pd.Series(array)
        return s.rank(ascending=False)[len(s)-1]

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before RollingRank is {df.shape}")
        #for col in self.columns:
        #    df[f'RRNK_{col}_{self.window}_{self.min_periods}'] = df[col].rolling(window=self.window,min_periods=self.min_periods).apply(self.rank)
            #pd.rolling_apply(df[col], window = self.window,min_periods=self.min_periods,func= self.rank)
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRNK_{col}_{self.window}_{self.min_periods}':df[col].rolling(window=self.window,min_periods=self.min_periods).apply(self.rank)})
            print_log(f"RRNK_{col}_{self.window}_{self.min_periods} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after RollingRank is {df.shape}") 
        return df

class BinningTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True):
        self.columns = columns
        self.get_current_row_bin = get_current_row_bin
        self.n_bins = n_bins
        self.verbose = verbose
        self.window = window
        self.min_period = min_period

    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario
    
    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before BinningTransform is {df.shape}")
        if self.get_current_row_bin:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[-1]
        else:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[0]
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'BINT_{col}_{self.window}_{self.min_period}_{self.n_bins}':df[col].rolling(window=self.window,min_periods=self.min_period).apply(bin_roll_fxn)})
            print_log(f"BINT_{col}_{self.window}_{self.min_period}_{self.n_bins} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after BinningTransform is {df.shape}") 
        return df

class PositiveNegativeTrends(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PositiveNegativeTrends is {df.shape}")
        #for col in self.columns:
        #    df[f'PNT_{col}_{self.window}_{self.min_periods}'] = df[col].pct_change().apply(np.sign).rolling(self.window, min_periods=self.min_periods).apply(np.sum)
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PNT_{col}_{self.window}_{self.min_periods}_DIFF':df[col].pct_change().apply(np.sign).rolling(self.window, min_periods=self.min_periods).apply(np.sum)})
            print_log(f"PNT_{col}_{self.window}_{self.min_periods}_DIFF completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after PositiveNegativeTrends is {df.shape}") 
        return df

class Rolling_Stats(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=30,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-np.array(x)[0])})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-max(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-min(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: np.array(x)[-1]-mean(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RR_{col}_{self.window}_{self.min_periods}_MAXMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lambda x: max(np.array(x))-min(np.array(x)))})
            print_log(f"RR_{col}_{self.window}_{self.min_periods}_MAXMIN completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats is {df.shape}") 
        return df

class Rolling_Stats_withLookBack(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=60,lookback_divider =2 ,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        self.lookback_divider = lookback_divider
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0]

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset]-np.array(vals)[0:offset+1].mean()

        def lookback_max_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals))-np.array(vals)[0:offset+1].mean()

        def lookback_min_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals))-np.array(vals)[0:offset+1].mean()

        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats_withLookBack is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_diff)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_mean)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MAXMAX':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max_max)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MAXMAX completed")
            merge_dict.update({f'RRLB_{col}_{self.window}_{self.min_periods}_MINMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min_min)})
            print_log(f"RRLB_{col}_{self.window}_{self.min_periods}_MINMIN completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats_withLookBack is {df.shape}") 
        return df

class Rolling_Stats_withLookBack_Compare(BaseEstimator, TransformerMixin):
    def __init__(self, columns,window=60,lookback_divider =2 ,min_periods=None,verbose=True):
        self.columns = columns
        self.window = window
        self.min_periods = min_periods
        self.verbose = verbose
        self.lookback_divider = lookback_divider
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        def lookback_diff(vals):
            offset = len(vals)//self.lookback_divider
            res = (np.array(vals)[offset]-np.array(vals)[-1]) - (np.array(vals)[0]-np.array(vals)[offset])
            return res

        def lookback_max(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        def lookback_min(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_mean(vals):
            offset = len(vals)//self.lookback_divider
            return np.array(vals)[offset+1:].mean()-np.array(vals)[0:offset+1].mean()

        def lookback_max_min(vals):
            offset = len(vals)//self.lookback_divider
            return max(np.array(vals)[offset+1:])-min(np.array(vals)[0:offset+1])

        def lookback_min_max(vals):
            offset = len(vals)//self.lookback_divider
            return min(np.array(vals)[offset+1:])-max(np.array(vals)[0:offset+1])

        if self.verbose:
            print_log(f"Shape of dataframe before Rolling_Stats_withLookBack_Compare is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_DIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_diff)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_DIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MAXDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MAXDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MINDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MINDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MEANDIFF':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_mean)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MEANDIFF completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MAXMIN':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_max_min)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MAXMIN completed")
            merge_dict.update({f'RRLBC_{col}_{self.window}_{self.min_periods}_MINMAX':df[col].rolling(self.window, min_periods=self.min_periods).apply(lookback_min_max)})
            print_log(f"RRLBC_{col}_{self.window}_{self.min_periods}_MINMAX completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after Rolling_Stats_withLookBack_Compare is {df.shape}") 
        return df

class PreviousDaysRange(BaseEstimator, TransformerMixin):
    def __init__(self, columns,freq='d',shift=1,resample='1min',verbose=True):
        '''
        freq should always be d or more than d. Any discontinuity in data would cause problem
        '''
        self.columns = columns
        self.freq = freq
        self.shift = shift
        self.resample = resample
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before PreviousDaysRange is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            merge_dict.update({f'PVDR_{col}_{self.freq}':df[col].resample(self.resample).ffill().groupby(pd.Grouper(freq=self.freq)).apply(lambda x: np.array(x)[-1]-np.array(x)[0]).shift(self.shift)})
            merge_dict.update({f'PVDR_{col}_{self.freq}_MAX_MIN':df[col].resample(self.resample).ffill().groupby(pd.Grouper(freq=self.freq)).apply(lambda x: max(np.array(x))-min(np.array(x))).shift(self.shift)})
            print_log(f"PVDR_{col}_{self.freq}_{self.shift} completed")
        df = pd.merge_asof(df, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        if self.verbose:
            print_log(f"Shape of dataframe after PreviousDaysRange is {df.shape}") 
        return df

class GapOpenMinuteChart(BaseEstimator, TransformerMixin):
    def __init__(self, columns,verbose=True):
        self.columns = columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before GapOpenMinuteChart is {df.shape}")
        merge_dict = {}
        for col in self.columns:
            tmp = df[col].resample('d').bfill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[0]).subtract( df[col].resample('d').ffill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[-1]).fillna(0))
            merge_dict.update({'GOMC_{col}':tmp[1:]})
            print_log(f"GOMC_{col} completed")
        df = pd.merge_asof(df, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        if self.verbose:
            print_log(f"Shape of dataframe after GapOpenMinuteChart is {df.shape}") 
        return df
class ConvertUnstableCols(BaseEstimator, TransformerMixin):
    def __init__(self,basis_column='close' ,tolerance=17000,verbose=True):
        self.basis_column = basis_column
        self.tolerance = tolerance
        self.verbose = verbose
        
    def fit(self, df, y=None):
        self.unstable_cols = [col for col in [c for c in df.columns.tolist() if df[c].dtypes != 'object'] if np.mean(df[col]) > 17000 ]
        return self     # Nothing to do in fit in this scenario

    def transform(self, df):
        print_log('*'*100)
        df = df.sort_index()
        if self.verbose:
            print_log(f"Shape of dataframe before ConvertUnstableCols is {df.shape}")
        merge_dict = {}
        print_log(f"Number of unstable columns are {len(self.unstable_cols)}")
        for col in self.unstable_cols:
            merge_dict.update({f'CUC_{col}': df[self.basis_column] - df[col]})
            #df[f'PerChg_{col}_{self.periods}_{self.freq}'] =df[col].pct_change(periods=self.periods,fill_method=self.fill_method,limit = self.limit,freq=self.freq)
            print_log(f"CUC_{col} completed")
        df = pd.concat([df,pd.concat(merge_dict,axis=1)],axis=1)
        if self.verbose:
            print_log(f"Shape of dataframe after ConvertUnstableCols is {df.shape}") 
        bt_pipe = Pipeline([
            ('bt1', BinningTransform(columns= self.unstable_cols,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt2', BinningTransform(columns= self.unstable_cols,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt3', BinningTransform(columns= self.unstable_cols,window=45,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt4', BinningTransform(columns= self.unstable_cols,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True))
                ])
        df = bt_pipe.fit_transform(df)
        df = df.drop(self.unstable_cols,axis=1)
        return df