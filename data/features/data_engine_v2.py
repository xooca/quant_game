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
import data.duckdb_utils as ddu

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
from config.common.config import Config,DefineConfig

def print_log(log,using_print=True):
    if using_print:
        print(log)
    else:
        logging.info(log)

def convert_todate_deduplicate(df):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')] 
    return df

class feature_mart(DefineConfig):
    def __init__(self,master_config_path,db_conection,verbose):
        DefineConfig.__init__(self,master_config_path)
        self.db_conection = db_conection
        self.ohlc = ddu.load_table_df(self.db_conection,table_name=self.ohlc_raw_data_table)
        self.feature_table_setup()
        self.verbose = verbose
        

    def feature_table_setup(self):
        sql = f"select max(timestamp) as max_timestamp from {self.ohlc_raw_data_table}"
        self.max_date_raw_table = self.db_conection.execute(sql).fetchone()[0]
        if not ddu.check_if_table_exists(self.db_conection,table_name=self.train_feature_table):
            df_empty = pd.DataFrame(columns=self.ohlc.columns.tolist())
            df_empty['timestamp']=pd.to_datetime(df_empty["timestamp"])
            ddu.create_table(self.db_conection,table_name=self.train_feature_table,create_table_arg={'replace':True},df=df_empty)
        try:
            sql = f"select max(timestamp) as max_timestamp from {self.train_feature_table}"
            self.max_date_feature_table = self.db_conection.execute(sql).fetchone()[0]
        except Exception as e:
            print_log(f"Error encountered while getting max date from {self.train_feature_table} : {e}") 
            self.max_date_feature_table = None
        if self.max_date_feature_table is not None:
            delta_df = self.ohlc[self.ohlc['timestamp']>self.max_date_feature_table]
            ddu.insert_data(self.db_conection,table_name=self.train_feature_table,insert_arg={},df=delta_df)
        else:
            ddu.insert_data(self.db_conection,table_name=self.train_feature_table,insert_arg={},df=self.ohlc)

    def convert_df_to_timeseries(self,df):
        df['date_time'] = df['date'].astype(str) + ' ' + df['time']
        df = df.sort_values(by='date_time')
        df.index = df['date_time']
        df = df[['open','high','low','close']]
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        return df

    def label_generator_14class(self,val):
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

    def label_generator_9class(self,val):
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

    def label_generator_7class(self,val):
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

    def label_generator_4class(self,val):
        if val <= 30 and val>=-30:
            return 'neutral'
        elif val > 30:
            return 'call'
        elif val < -30:
            return 'put'
        else:
            return 'unknown'

    def create_column_and_save_to_table(self,time_stamp_col,data):
        column_names = [col for col in data.columns.tolist() if col != time_stamp_col]
        for column_name in column_names:
            if not ddu.check_if_table_and_column_exists(self.db_conection,self.train_feature_table,column_name):
                alter_arg = {'alter_type':'add_column','column_name':column_name,'data_type':'DOUBLE'}
                ddu.alter_table(self.db_conection,self.train_feature_table,alter_arg=alter_arg)
                #update_arg = {'column_name':column_name,'set_expr':set_expr,'where_expr':}
            sql = f'''insert into {self.train_feature_table} ({column_name}) 
            select tab2.{column_name} from {self.train_feature_table} as tab1 inner join data as tab2
            on tab1.{time_stamp_col} = tab2.{time_stamp_col}
            '''
            print(f"SQL to be run is {sql}")
            self.db_conection.execute(sql)

    def get_ohlc_df(self,df=None):
        if df is None:
            df = self.ohlc_df.copy()
        df.index = df['timestamp']
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df

    def remove_ohlc_cols(self,df):
        ohlc_cols = ['open','close','high','low']
        all_cols = [col for col in df.columns.tolist() if col not in ohlc_cols]
        return df[all_cols]

    def keep_timestamp_feature_cols(self,df,cols):
        final_cols = ['timestamp'] + cols
        return df[final_cols]

    def label_creator(self,func_dict_args,tmpdf=None):
        # freq='1min',shift=-15,shift_column='close',generator_function_name='label_generator_4class'
        print_log(f"label_creator called with arguments {func_dict_args}") 
        label_name = f"label_{func_dict_args['shift']}_{func_dict_args['freq']}_{func_dict_args['shift_column']}__{func_dict_args['generator_function_name']}"
        if tmpdf is None:
            tmpdf = self.get_ohlc_df()
        tmpdf[label_name] = tmpdf.shift(func_dict_args['shift'], freq=func_dict_args['freq'])[func_dict_args['shift_column']].subtract(tmpdf[func_dict_args['shift_column']]).apply(func_dict_args['generator_function_name']) 
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def create_technical_indicator_using_pandasta(self,func_dict_args):
        # ohlc_df,exclude=["pvo","vwap","vwma","ad","adosc","aobv","cmf","efi","eom","kvo","mfi","nvi","obv","pvi","pvol","pvr","pvt"]
        import pandas_ta as ta
        print_log(f"create_technical_indicator_using_pandasta called with arguments {func_dict_args}") 
        tmpdf = self.get_ohlc_df()
        tmpdf.ta.strategy(exclude=func_dict_args['exclude'],verbose=self.verbose,timed=True)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def create_technical_indicator_using_signals(self,func_dict_args):
        #method_type = ['volumn_','volatile_','transform_','cycle_','pattern_','stats_','math_','overlap_']
        import pandas_ta as ta
        print_log(f"create_technical_indicator_using_signals called with arguments {func_dict_args}") 
        tmpdf = self.get_ohlc_df()
        
        all_methods = []
        a = dict(Signals.__dict__)
        for a1,a2 in a.items():
            all_methods.append(a1)
        all_methods = [m1 for m1,m2 in a.items() if m1[:1]!='_']
        all_methods = [m for m in all_methods for mt in func_dict_args.get('method_type') if mt in m]

        sig = Signals(tmpdf)
        methods_run = []
        methods_notrun = []
        for f in self.all_methods:
            try:
                exec(f'sig.{f}()')
                methods_run.append(f)
            except Exception as e1:
                print_log(f"Function {f} was unable to run, Error is {e1}")
                methods_notrun.append(f)

        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def create_technical_indicator_using_ta(self,func_dict_args):
        #open='open',high='high',low='low',close='close',volume='volume',vectorized=True,fillna=False,colprefix='ta',volume_ta=True,volatility_ta=True,trend_ta=True,momentum_ta=True,others_ta=True,verbose=True
        print_log(f"create_technical_indicator_using_ta called with arguments {func_dict_args}") 
        tmpdf = self.get_ohlc_df()
        tmpdf = add_all_ta_features(
            tmpdf,
            open = func_dict_args.get('open'),
            high = func_dict_args.get('high'),
            low = func_dict_args.get('low'),
            close = func_dict_args.get('close'),
            volume = func_dict_args.get('volume'),
            fillna = func_dict_args.get('fillna'),
            colprefix = func_dict_args.get('colprefix'),
            vectorized = func_dict_args.get('vectorized'),
            volume_ta = func_dict_args.get('volume_ta'),
            volatility_ta  = func_dict_args.get('volatility_ta'),
            trend_ta  = func_dict_args.get('trend_ta'),
            momentum_ta  = func_dict_args.get('momentum_ta'),
            others_ta = func_dict_args.get('others_ta'),
        )
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def normalize_dataset(self,df,func_dict_args):
        #  column_pattern = [],columns = [],impute_values=False,impute_type = 'categorical',convert_to_floats = False,arbitrary_impute_variable=99,drop_na_col=False,drop_na_rows=False,
        #  fillna = False,fillna_method = 'bfill',fill_index=False
        print_log(f"normalize_dataset called with arguments {func_dict_args}") 
        if len(func_dict_args.get('columns')) == 0:
            func_dict_args['columns'] = [m for m in df.columns.tolist() for mt in func_dict_args.get('column_pattern') if mt in m]
            func_dict_args['columns'] = list(set(func_dict_args.get('columns')))

        info_list = []
        df = convert_todate_deduplicate(df)
        if func_dict_args.get('convert_to_floats'):
            for col in func_dict_args['columns']:
                df[col] = df[col].astype('float')
                info_list.append('convert_to_floats')
        if func_dict_args.get('fill_index'):
            df = df.reindex(pd.date_range(min(df.index), max(df.index), freq ='1min'))
            df = df.resample('1min').ffill()
        if func_dict_args.get('impute_values'):
            from sklearn.pipeline import Pipeline
            if func_dict_args.get('impute_type') == 'mean_median_imputer':
                imputer = MeanMedianImputer(imputation_method='median', variables=func_dict_args['columns'])
                info_list.append('mean_median_imputer')
            elif func_dict_args.get('impute_type') == 'categorical':
                imputer = CategoricalImputer(variables=func_dict_args['columns'])
                info_list.append('categorical')
            elif func_dict_args.get('impute_type') == 'arbitrary':
                if isinstance(func_dict_args.get('arbitrary_impute_variable'), dict):
                    imputer = ArbitraryNumberImputer(imputer_dict = func_dict_args.get('arbitrary_impute_variable'))  
                else:
                    imputer = ArbitraryNumberImputer(variables = func_dict_args['columns'],arbitrary_number = func_dict_args.get('arbitrary_number'))
                info_list.append('arbitrary')
            else:
                imputer = CategoricalImputer(variables=func_dict_args['columns'])
                info_list.append('categorical')
            imputer.fit(df)
            df= imputer.transform(df)
        if func_dict_args.get('fillna'):
            df = df.fillna(method=self.fillna_method)
            info_list.append('fillna')
        if func_dict_args.get('drop_na_col'):
            imputer = DropMissingData(missing_only=True)
            imputer.fit(df)
            df= imputer.transform(df)
            info_list.append('drop_na_col')
        if func_dict_args.get('drop_na_rows'):
            df = df.dropna(axis=0)
            info_list.append('drop_na_rows')
        df = df.sort_index()
        return df

    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def last_tick_greater_values_count(self,func_dict_args):
        # column,last_ticks=10
        print_log(f"last_tick_greater_values_count called with arguments {func_dict_args}") 
        feature_name = f"ROLLTGC_{func_dict_args.get('column')}_{func_dict_args.get('last_ticks')}"
        tmpdf = self.get_ohlc_df()
        x = np.concatenate([[np.nan] * (func_dict_args.get('last_ticks')), tmpdf[func_dict_args.get('column')].values])
        arr = self.rolling_window(x, func_dict_args.get('last_ticks') + 1)
        tmpdf[feature_name]  = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
        tmpdf = self.keep_timestamp_feature_cols(tmpdf,cols=[feature_name])
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def price_last_tick_breach_count(self,func_dict_args):
        # column_name=[],last_ticks='5',breach_type = ['morethan']
        print_log(f"price_last_tick_breach_count called with arguments {func_dict_args}") 
        tmpdf = self.get_ohlc_df()
        feature_name = f"ROLLTBC_{func_dict_args.get('breach_type')}_{func_dict_args.get('column_name')}_{self.last_ticks}"
        if func_dict_args.get('breach_type') == 'morethan':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
        elif func_dict_args.get('breach_type') == 'lessthan':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
        elif func_dict_args.get('breach_type') == 'mean':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == 'min':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == 'max':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == 'median':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == '10thquantile':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == '25thquantile':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == '75thquantile':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
        elif func_dict_args.get('breach_type') == '95thquantile':
            tmpdf[feature_name] = tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
        else:
            tmpdf[feature_name] = (tmpdf[func_dict_args.get('column_name')].rolling(self.last_ticks, min_periods=1)
                    .apply(lambda x: (x[-1] > x[:-1]).mean())
                    .astype(int))
        tmpdf = self.keep_timestamp_feature_cols(tmpdf,cols=[feature_name])
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def rolling_values(self,func_dict_args):
        # columns=[],last_ticks=['5min','10min'],aggs=['mean','max'],oper = ['-','=']
        print_log(f"rolling_values called with arguments {func_dict_args}") 
        column = func_dict_args.get('column')
        last_ticks = func_dict_args.get('last_ticks')
        aggs = func_dict_args.get('aggs')
        oper = func_dict_args.get('oper')
        tmpdf = self.get_ohlc_df()

        eval_stmt = '' 
        for lt,oper,agg in zip(last_ticks,oper,aggs):
            tmpst = f"tmpdf[{column}].rolling('{lt}', min_periods=1).{agg}() {oper}"
            eval_stmt = eval_stmt + tmpst
        col_name = f"ROLLVAL{column}_{'_'.join(last_ticks)}_{'_'.join(aggs)}"
        tmpdf[col_name] = eval(eval_stmt[:-1])
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def price_data_range_hour(self,func_dict_args):
        # first_col = 'high',second_col='low',hour_range = [['09:00', '10:30'],['10:30', '11:30']],range_type=['price_range','price_deviation_max_first_col']
        print_log(f"price_data_range_hour called with arguments {func_dict_args}") 
        r1 = func_dict_args.get('hour_range')[0]
        r2 = func_dict_args.get('hour_range')[1]
        first_col = func_dict_args.get('first_col')
        second_col = func_dict_args.get('second_col')
        range_type = func_dict_args.get('range_type')
        col_name = f"ROLLPDR_{first_col}_{second_col}_{range_type}_{r1.replace(':','')}_{r2.replace(':','')}"
        tmpdf = self.get_ohlc_df()
        if range_type == 'price_range':
            tmpdf[col_name]  = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        elif range_type == 'price_deviation_max_first_col':
            tmpdf[col_name]  = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
        elif range_type == 'price_deviation_min_first_col':
            tmpdf[col_name]  = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        elif range_type == 'price_deviation_max_second_col':
            tmpdf[col_name]  = tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
        elif range_type == 'price_deviation_min_second_col':
            tmpdf[col_name]  = tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        else:
            tmpdf[col_name]  = tmpdf[first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - tmpdf[second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
        tmpdf[col_name] = tmpdf[col_name].fillna(method='ffill')
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def price_velocity_v2(self,func_dict_args):
        #freq='D',shift=5,shift_column=['close','open']
        print_log(f"price_velocity called with arguments {func_dict_args}") 
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')
        tmpdf = self.get_ohlc_df()
        if freq is not None:
            col_name = f'ROLLPVR2_{shift_column}_{freq}_{shift}'
            tmpdf[col_name] = tmpdf.shift(shift, freq=self.freq)[shift_column]
        else:
            col_name = f'ROLLPVR2_{shift_column}_{shift}'
            tmpdf[col_name] = tmpdf.shift(shift)[shift_column]
        tmpdf[col_name] = tmpdf[shift_column] - tmpdf[col_name]
        tmpdf[col_name] = tmpdf[col_name].round(3)
        print_log(f"price_velocity : {col_name} created")
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def price_velocity(self,func_dict_args):
        #freq='D',shift=5,shift_column=['close','open']
        print_log(f"price_velocity called with arguments {func_dict_args}") 
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')
        tmpdf = self.get_ohlc_df()
        if freq is not None:
            col_name = f'ROLLPVC_{shift_column}_{freq}_{shift}'
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift,freq=freq)[shift_column])
        else:
            col_name = f'ROLLPVC_{shift_column}_{shift}'
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift)[shift_column])
        tmpdf[col_name] = tmpdf[col_name].round(3)
        print_log(f"price_velocity : {col_name} created")
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def price_velocity_rate(self,func_dict_args):
        #freq='D',shift=5,shift_column=['close','open']
        print_log(f"price_velocity called with arguments {func_dict_args}") 
        freq = func_dict_args.get('freq')
        shift = func_dict_args.get('shift')
        shift_column = func_dict_args.get('shift_column')
        tmpdf = self.get_ohlc_df()
        if freq is not None:
            col_name = f'ROLLPVR_{shift_column}_{freq}_{shift}'
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift,freq=freq)[shift_column])/shift
        else:
            col_name = f'ROLLPVR_{shift_column}_{shift}'
            tmpdf[col_name] = tmpdf[shift_column].subtract(tmpdf.shift(shift)[shift_column])/shift
        tmpdf[col_name] = tmpdf[col_name].round(3)
        print_log(f"price_velocity : {col_name} created")
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def filter_data(self,func_dict_args,df=None):
        # start_date=None,end_date=None,filter_rows=None,

        print_log(f"filter_data called with arguments {func_dict_args}") 
        start_date = func_dict_args.get('start_date')
        end_date = func_dict_args.get('end_date')
        filter_rows = func_dict_args.get('filter_rows')
        if df is None:
            df = self.get_ohlc_df()
        print_log(f"Shape of dataframe before FilterData is {df.shape}") 
        if (start_date != 'None' and end_date == 'None') or (start_date is not None and end_date is None ):
            df = df.sort_index().loc[start_date:]
            print_log(f"Data filtered with {start_date}") 
        elif (start_date == 'None' and end_date != 'None') or (start_date is None and end_date is not None):
            df = df.sort_index().loc[:end_date]
            print_log(f"Data filtered with {end_date}") 
        elif (start_date != 'None' and end_date != 'None') or (start_date is not None and end_date is not None):
            df = df.sort_index().loc[start_date:end_date]
            print_log(f"Data filtered with {end_date}") 
        else:
            df = df.sort_index()
            print_log(f"No filtering done") 
        if filter_rows != 'None' or filter_rows is not None:
            df = df[:filter_rows]
            print_log(f"Data filtered with filter rows {filter_rows}") 
        if self.verbose:
            print_log(f"Shape of dataframe after FilterData is {df.shape}") 
        return df

    def zscore(self,x, window):
        r = x.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (x-m)/s
        return z

    def rolling_zscore(self,func_dict_args):
        # columns,window = 30
        print_log(f"rolling_zscore called with arguments {func_dict_args}") 
        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLZSR_{column}_{window}':self.zscore(tmpdf[column],window)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf

    def rolling_log_transform(self,func_dict_args):
        # column
        print_log(f"rolling_zscore called with arguments {func_dict_args}") 
        column = func_dict_args.get('column')
        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLLOG_{column}':tmpdf[column].apply(np.log)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_percentage_change(self,func_dict_args):
        # columns,periods=30, fill_method='pad', limit=None, freq=None,verbose=False
        print_log(f"rolling_percentage_change called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        periods = func_dict_args.get('periods')
        fill_method = func_dict_args.get('fill_method')
        limit = func_dict_args.get('limit')
        freq = func_dict_args.get('freq')

        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLPCH_{column}_{periods}_{freq}':tmpdf[column].pct_change(periods=periods,fill_method=fill_method,limit = limit,freq=freq)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_percentage_change_multiplier(self,func_dict_args):
        # columns,periods=30, fill_method='pad', limit=None, freq=None,multiplier=100
        print_log(f"rolling_percentage_change_multiplier called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        periods = func_dict_args.get('periods')
        fill_method = func_dict_args.get('fill_method')
        limit = func_dict_args.get('limit')
        freq = func_dict_args.get('freq')
        multiplier = func_dict_args.get('multiplier')

        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLPCM_{column}_{periods}_{freq}':tmpdf[column].pct_change(periods=periods,fill_method=fill_method,limit = limit,freq=freq)*multiplier})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_weighted_exponential_average(self,func_dict_args):
        # columns,com=None, span=44, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None
        print_log(f"rolling_weighted_exponential_average called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        com = func_dict_args.get('com')
        span = func_dict_args.get('span')
        halflife = func_dict_args.get('halflife')
        alpha = func_dict_args.get('alpha')
        min_periods = func_dict_args.get('min_periods')
        adjust = func_dict_args.get('adjust')
        ignore_na = func_dict_args.get('ignore_na')
        axis = func_dict_args.get('axis')
        times = func_dict_args.get('times')

        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLWEA_{column}_{span}':tmpdf[column].ewm(com=com, span=span, halflife=halflife, alpha=alpha, min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis, times=times).mean()})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_percentile(self,func_dict_args):
        # columns,window=30,min_periods=None,quantile=0.75
        print_log(f"rolling_percentile called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        quantile = func_dict_args.get('quantile')

        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLPCT_{column}_{window}_{min_periods}':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: np.array(x)[-1] - np.quantile(np.array(x),q=quantile))})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_rank(self,func_dict_args):
        # columns,window=30,min_periods=None
        print_log(f"rolling_rank called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')

        tmpdf = self.get_ohlc_df()
        merge_dict = {}
        merge_dict.update({f'ROLLRNK_{column}_{window}_{min_periods}':tmpdf[column].rolling(window=window,min_periods=min_periods).apply(rank)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_binning(self,func_dict_args):
        # column,window=30,min_periods=None,get_current_row_bin=True,n_bins=5
        print_log(f"rolling_binning called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        get_current_row_bin = func_dict_args.get('get_current_row_bin')
        n_bins = func_dict_args.get('n_bins')

        tmpdf = self.get_ohlc_df()
        if get_current_row_bin:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[-1]
        else:
            bin_roll_fxn = lambda x: pd.qcut(np.array(x),labels=False,q=self.n_bins,duplicates='drop')[0]
        merge_dict = {}
        merge_dict.update({f'ROLLBIN_{column}_{window}_{min_periods}_{n_bins}':tmpdf[column].rolling(window=window,min_periods=min_periods).apply(bin_roll_fxn)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_trends(self,func_dict_args):
        # column,window=30,min_period=None,get_current_row_bin=True,n_bins=5
        print_log(f"rolling_trends called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')

        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        merge_dict.update({f'ROLLTRN_{column}_{window}_{min_periods}':tmpdf[column].pct_change().apply(np.sign).rolling(window, min_periods=min_periods).apply(np.sum)})
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_stats(self,func_dict_args):
        # columns,window=30,min_periods=None
        print_log(f"rolling_stats called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')

        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        merge_dict.update({f'ROLLSTT_{column}_{window}_{min_periods}_DIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-np.array(x)[0])})
        print_log(f"ROLLSTT_{column}_{window}_{min_periods}_DIFF completed")
        merge_dict.update({f'ROLLSTT_{column}_{window}_{min_periods}_MAXDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-max(np.array(x)))})
        print_log(f"ROLLSTT_{column}_{window}_{min_periods}_MAXDIFF completed")
        merge_dict.update({f'ROLLSTT_{column}_{window}_{min_periods}_MINDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-min(np.array(x)))})
        print_log(f"ROLLSTT_{column}_{window}_{min_periods}_MINDIFF completed")
        merge_dict.update({f'ROLLSTT_{column}_{window}_{min_periods}_MEANDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: np.array(x)[-1]-mean(np.array(x)))})
        print_log(f"ROLLSTT_{column}_{window}_{min_periods}_MEANDIFF completed")
        merge_dict.update({f'ROLLSTT_{column}_{window}_{min_periods}_MAXMIN':tmpdf[column].rolling(window, min_periods=min_periods).apply(lambda x: max(np.array(x))-min(np.array(x)))})
        print_log(f"ROLLSTT_{column}_{window}_{min_periods}_MAXMIN completed")
        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_stats_lookback(self,func_dict_args):
        # columns,window=60,lookback_divider =2,min_periods=None
        print_log(f"rolling_stats_lookback called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.lookback_divider = func_dict_args.get('lookback_divider')

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


        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_diff)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF completed")
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF completed")
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF completed")
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF completed")
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMAX':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_max)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMAX completed")
        merge_dict.update({f'ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMIN':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_min)})
        print_log(f"ROLLSLB_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMIN completed")

        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_stats_lookback_compare(self,func_dict_args):
        # columns,window=60,lookback_divider =2 ,min_periods=None
        print_log(f"rolling_stats_lookback_compare called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        window = func_dict_args.get('window')
        min_periods = func_dict_args.get('min_periods')
        self.lookback_divider = func_dict_args.get('lookback_divider')

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


        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_DIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_diff)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_DIFF completed")
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXDIFF completed")
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINDIFF completed")
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_mean)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MEANDIFF completed")
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMIN':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_max_min)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MAXMIN completed")
        merge_dict.update({f'ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX':tmpdf[column].rolling(window, min_periods=min_periods).apply(lookback_min_max)})
        print_log(f"ROLLLBC_{column}_{window}_{min_periods}_{self.lookback_divider}_MINMAX completed")

        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_previous_day_range(self,func_dict_args):
        # columns,freq='d',shift_val=1,resample='1min'
        print_log(f"rolling_previous_day_range called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')
        freq = func_dict_args.get('freq')
        shift_val = func_dict_args.get('shift_val')
        resample = func_dict_args.get('resample')

        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        merge_dict.update({f'ROLLPDR_{column}_{freq}_{shift_val}_{resample}':tmpdf[column].resample(resample).ffill().groupby(pd.Grouper(freq=freq)).apply(lambda x: np.array(x)[-1]-np.array(x)[0]).shift(shift_val)})
        merge_dict.update({f'ROLLPDR_{column}_{freq}_{shift_val}_{resample}_MAX_MIN':tmpdf[column].resample(resample).ffill().groupby(pd.Grouper(freq=freq)).apply(lambda x: max(np.array(x))-min(np.array(x))).shift(shift_val)})

        tmpdf = pd.concat([tmpdf,pd.concat(merge_dict,axis=1)],axis=1)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def rolling_gap_open_min(self,func_dict_args):
        # column
        print_log(f"rolling_previous_day_range called with arguments {func_dict_args}") 

        column = func_dict_args.get('column')

        tmpdf = self.get_ohlc_df()

        merge_dict = {}
        a = tmpdf[column].resample('d').bfill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[0]).subtract( tmpdf[column].resample('d').ffill().groupby(pd.Grouper(freq='d')).apply(lambda x:x[-1]).fillna(0))
        merge_dict.update({'ROLLGOM_{column}':a[1:]})        

        tmpdf = pd.merge_asof(tmpdf, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        tmpdf = self.remove_ohlc_cols(tmpdf)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

    def treat_unstable_cols(self,df,func_dict_args):
        # basis_column='close',ohlc_columns = ['close','open','high','low'],tolerance=17000,transform_option='rank'
        print_log(f"rolling_previous_day_range called with arguments {func_dict_args}") 

        basis_column = func_dict_args.get('basis_column')
        ohlc_columns = func_dict_args.get('ohlc_columns')
        tolerance = func_dict_args.get('tolerance')
        transform_option = func_dict_args.get('transform_option')

        unstable_cols = [col for col in [c for c in df.columns.tolist() if df[c].dtypes != 'object'] if np.mean(df[col]) > tolerance ]
        unstable_cols = [col for col in unstable_cols if col not in ohlc_columns]

        if transform_option == 'bint':
            for col in unstable_cols:
                func_dict_args = {'column':col,'window':15,'min_periods':None,'get_current_row_bin':True,'n_bins':5}
                self.rolling_binning(func_dict_args)
                func_dict_args = {'column':col,'window':30,'min_periods':None,'get_current_row_bin':True,'n_bins':5}
                self.rolling_binning(func_dict_args)
                func_dict_args = {'column':col,'window':45,'min_periods':None,'get_current_row_bin':True,'n_bins':5}
                self.rolling_binning(func_dict_args)
                func_dict_args = {'column':col,'window':60,'min_periods':None,'get_current_row_bin':True,'n_bins':5}
                self.rolling_binning(func_dict_args)
        elif transform_option == 'rank':
            for col in unstable_cols:
                func_dict_args = {'column':col,'window':15,'min_periods':None}
                self.rolling_rank(func_dict_args)
                func_dict_args = {'column':col,'window':15,'min_periods':None}
                self.rolling_binning(func_dict_args)
                func_dict_args = {'column':col,'window':15,'min_periods':None}
                self.rolling_binning(func_dict_args)
                func_dict_args = {'column':col,'window':15,'min_periods':None}                
                self.rolling_binning(func_dict_args)
        else:


        def rolling_percentage_change_multiplier(self,func_dict_args):
        # columns,periods=30, fill_method='pad', limit=None, freq=None,multiplier=100
            
            self.rolling_binning()
            bt_pipe = Pipeline([
                ('bt1', BinningTransform(columns= self.unstable_cols,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt2', BinningTransform(columns= self.unstable_cols,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt3', BinningTransform(columns= self.unstable_cols,window=45,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt4', BinningTransform(columns= self.unstable_cols,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True))
                    ])
        elif transform_option == 'rank':
            bt_pipe = Pipeline([
                ('bt1', RollingRank(columns= self.unstable_cols,window=15,min_periods=None,verbose=True)),
                ('bt2', RollingRank(columns= self.unstable_cols,window=30,min_periods=None,verbose=True)),
                ('bt3', RollingRank(columns= self.unstable_cols,window=45,min_periods=None,verbose=True)),
                ('bt4', RollingRank(columns= self.unstable_cols,window=60,min_periods=None,verbose=True))
                    ])
        else:
            bt_pipe = Pipeline([
                ('bt1', PercentageChange_Multiplier(columns= self.unstable_cols,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt2', PercentageChange_Multiplier(columns= self.unstable_cols,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt3', PercentageChange_Multiplier(columns= self.unstable_cols,periods=45, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt4', PercentageChange_Multiplier(columns= self.unstable_cols,periods=60, fill_method='pad', limit=None, freq=None,verbose=True))
                    ])
     
        tmpdf = pd.merge_asof(tmpdf, pd.concat(merge_dict,axis=1), left_index=True, right_index=True)
        self.create_column_and_save_to_table(time_stamp_col='timestamp',data = tmpdf)
        del tmpdf,merge_dict

class ConvertUnstableCols(BaseEstimator, TransformerMixin):
    def __init__(self,basis_column='close',ohlc_columns = ['close','open','high','low'],tolerance=17000,transform_option='rank',verbose=True):
        self.basis_column = basis_column
        self.tolerance = tolerance
        self.transform_option = transform_option
        self.ohlc_columns = ohlc_columns
        self.verbose = verbose
        
    def fit(self, df, y=None):
        self.unstable_cols = [col for col in [c for c in df.columns.tolist() if df[c].dtypes != 'object'] if np.mean(df[col]) > self.tolerance ]
        self.unstable_cols = [col for col in self.unstable_cols if col not in self.ohlc_columns]
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
        if self.transform_option == 'bint':
            bt_pipe = Pipeline([
                ('bt1', BinningTransform(columns= self.unstable_cols,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt2', BinningTransform(columns= self.unstable_cols,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt3', BinningTransform(columns= self.unstable_cols,window=45,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ('bt4', BinningTransform(columns= self.unstable_cols,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True))
                    ])
        elif self.transform_option == 'rank':
            bt_pipe = Pipeline([
                ('bt1', RollingRank(columns= self.unstable_cols,window=15,min_periods=None,verbose=True)),
                ('bt2', RollingRank(columns= self.unstable_cols,window=30,min_periods=None,verbose=True)),
                ('bt3', RollingRank(columns= self.unstable_cols,window=45,min_periods=None,verbose=True)),
                ('bt4', RollingRank(columns= self.unstable_cols,window=60,min_periods=None,verbose=True))
                    ])
        else:
            bt_pipe = Pipeline([
                ('bt1', PercentageChange_Multiplier(columns= self.unstable_cols,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt2', PercentageChange_Multiplier(columns= self.unstable_cols,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt3', PercentageChange_Multiplier(columns= self.unstable_cols,periods=45, fill_method='pad', limit=None, freq=None,verbose=True)),
                ('bt4', PercentageChange_Multiplier(columns= self.unstable_cols,periods=60, fill_method='pad', limit=None, freq=None,verbose=True))
                    ])
        df = bt_pipe.fit_transform(df)
        print_log(f"Shape of dataframe after applying transform in ConvertUnstableCols is {df.shape}") 
        df = df.drop(self.unstable_cols,axis=1)
        print_log(f"Shape of dataframe after dropping unstable columns is {df.shape}")
        return df