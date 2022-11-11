
from sklearn.pipeline import Pipeline
import pickle
import logging
import data.data_engine as de
import pandas as pd
import numpy as np
#import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import data.data_utils_old as du

from config.common.config import Config

class CustomConfig(Config):
    def initialize_all_config(self):
        super(Config, self).initialize_all_config() 
        self.pycaret_setup = self.config.trainer.setup
class pipelines:
    def __init__(self,dc):
        self.dc = dc
        self.OHLC_COLUMNS = list(dc.data.common.ohlc_column)
        self.TECHNICAL_IND_PATTERN = list(dc.data.common.technical_indicator_col_pattern)
        self.SELECTED_COLUMNS = list(dc.data.common.selected_columns)
        self.TA_PIPE2_EXCLUDE = list(dc.data.common.ta_pipe2_exclude)
        self.TA_BASIS_COL = dc.data.tech_ind.basis_column
        self.TA_TOLERANCE = dc.data.tech_ind.tolerance
    
    def pipeline_definitions(self):
        self.start_pipeline = Pipeline([
                            ('nd_gen', de.NormalizeDataset(columns = self.OHLC_COLUMNS)),
                            ('read_fd', de.FilterData(start_date=self.dc.data.common.read_start_date,end_date=self.dc.data.common.read_end_date)),
                            ])
                            
        self.generic_pipeline = Pipeline([
                            ('nd_gen', de.NormalizeDataset(columns = self.OHLC_COLUMNS)),
                            ('read_fd', de.FilterData(start_date=self.dc.data.common.read_start_date,end_date=self.dc.data.common.read_end_date)),
                            ])

        self.save_pipeline = Pipeline([
                            ('save_fd', de.FilterData(start_date=self.dc.data.common.save_start_date,end_date=self.dc.data.common.save_end_date)),
                            ])

        self.technical_ind_pre_pipe = Pipeline([
            ('tech_ind_pre_ND', de.NormalizeDataset(columns = self.OHLC_COLUMNS,impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
            ])
            
        self.technical_indicator_pipe = Pipeline([
            ('tech_ind_pre_ND', de.NormalizeDataset(columns = self.OHLC_COLUMNS,impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
            ('tech_indicator1', de.TechnicalIndicator(method_type = self.TECHNICAL_IND_PATTERN)),
            ('tech_indicator1_ND1', de.NormalizeDataset(column_pattern = self.OHLC_COLUMNS + self.TECHNICAL_IND_PATTERN,fillna=True,fillna_method='bfill')),
            ('tech_indicator1_ND2', de.NormalizeDataset(column_pattern = self.OHLC_COLUMNS + self.TECHNICAL_IND_PATTERN,fillna=True,fillna_method='ffill')),
            ('tech_indicator2', de.CreateTechnicalIndicatorUsingPandasTA(exclude=self.TA_PIPE2_EXCLUDE,verbose=True)),
            ('tech_indicator3', de.CreateTechnicalIndicatorUsingTA(volume_ta=False,verbose=True)),
            ('tech_indicator4', de.ConvertUnstableCols(basis_column=self.TA_BASIS_COL ,ohlc_columns = self.OHLC_COLUMNS,tolerance=self.TA_TOLERANCE,transform_option='others',verbose=True)),
            ])


        self.rolling_values_pipe = Pipeline([
            ('rv1', de.RollingValues(columns= self.SELECTED_COLUMNS,column_pattern=[],last_ticks=['10min','30min'],aggs=['mean','mean'],oper = ['-','='],verbose=True)),
            ('rv2', de.RollingValues(columns= self.SELECTED_COLUMNS,column_pattern=[],last_ticks=['10min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
            ('rv3', de.RollingValues(columns= self.SELECTED_COLUMNS,column_pattern=[],last_ticks=['5min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
            ('rv4', de.RollingValues(columns= self.SELECTED_COLUMNS,column_pattern=[],last_ticks=['15min','60min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
            ])

        self.last_tick_greater_values_pipe = Pipeline([    
            ('ltgvc1', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=10)),
            ('ltgvc2', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=15)),
            ('ltgvc3', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=30)),  
            ('ltgvc4', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=60)), 
            ('ltgvc5', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=90)), 
            ('ltgvc6', de.LastTicksGreaterValuesCount(column_pattern=[],columns= self.SELECTED_COLUMNS,create_new_col = True,last_ticks=120)),  
            ])

        self.zscore_log_percentage_chg_pipe = Pipeline([
            ('zs1', de.Zscoring(columns=self.SELECTED_COLUMNS,window=10,verbose=True)),
            ('zs2', de.Zscoring(columns=self.SELECTED_COLUMNS,window=15,verbose=True)),
            ('zs3', de.Zscoring(columns=self.SELECTED_COLUMNS,window=30,verbose=True)),
            ('zs4', de.Zscoring(columns=self.SELECTED_COLUMNS,window=60,verbose=True)),

            ('lgt', de.LogTransform(columns=self.SELECTED_COLUMNS,verbose=True)),

            ('pc1', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=10, fill_method='pad', limit=None, freq=None,verbose=True)),
            ('pc2', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
            ('pc3', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
            ('pc4', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=60, fill_method='pad', limit=None, freq=None,verbose=True)),
            ('pc5', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=90, fill_method='pad', limit=None, freq=None,verbose=True)),
            ('pc6', de.PercentageChange(columns=self.SELECTED_COLUMNS,periods=120, fill_method='pad', limit=None, freq=None,verbose=True)),
                ])

## percentile_transform_pipe not included because it takes lot of time
        self.percentile_transform_pipe = Pipeline([
            ('pt1', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=10,min_periods=None,quantile=0.75,verbose=True)),
            ('pt2', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=15,min_periods=None,quantile=0.75,verbose=True)),
            ('pt3', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=30,min_periods=None,quantile=0.75,verbose=True)),
            ('pt4', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=60,min_periods=None,quantile=0.75,verbose=True)),
            ('pt5', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=90,min_periods=None,quantile=0.75,verbose=True)),
            ('pt6', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=120,min_periods=None,quantile=0.75,verbose=True)),
                ])

## percentile_transform_pipe1 not included because it takes lot of time
        self.percentile_transform_pipe1 = Pipeline([
            ('pt11', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=10,min_periods=None,quantile=0.90,verbose=True)),
            ('pt22', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=15,min_periods=None,quantile=0.90,verbose=True)),
            ('pt33', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=30,min_periods=None,quantile=0.90,verbose=True)),
            ('pt44', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=60,min_periods=None,quantile=0.90,verbose=True)),
            ('pt55', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=90,min_periods=None,quantile=0.90,verbose=True)),
            ('pt66', de.PercentileTransform(columns= self.SELECTED_COLUMNS,window=120,min_periods=None,quantile=0.90,verbose=True)),
                ])

        self.rolling_rank_pipe = Pipeline([
            ('rr1', de.RollingRank(columns= self.SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
            ('rr2', de.RollingRank(columns= self.SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
            ('rr3', de.RollingRank(columns= self.SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
            ('rr4', de.RollingRank(columns= self.SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),
            ('rr5', de.RollingRank(columns= self.SELECTED_COLUMNS,window=90,min_periods=None,verbose=True)),
            ('rr6', de.RollingRank(columns= self.SELECTED_COLUMNS,window=120,min_periods=None,verbose=True)),
                ])

        self.bin_transform_pipe = Pipeline([
            ('bt1', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=10,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt2', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt3', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt4', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt5', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=90,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
            ('bt6', de.BinningTransform(columns= self.SELECTED_COLUMNS,window=120,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
                ])

        self.positive_negative_pipe = Pipeline([
            ('pnt1', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
            ('pnt2', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
            ('pnt3', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
            ('pnt4', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),
            ('pnt5', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=90,min_periods=None,verbose=True)),
            ('pnt6', de.PositiveNegativeTrends(columns= self.SELECTED_COLUMNS,window=120,min_periods=None,verbose=True)),
                ])

        self.rolling_stats_pipe = Pipeline([
            ('rs1', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
            ('rs2', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
            ('rs3', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
            ('rs4', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),
            ('rs5', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=90,min_periods=None,verbose=True)),
            ('rs6', de.Rolling_Stats(columns= self.SELECTED_COLUMNS,window=120,min_periods=None,verbose=True)),

            ('rswl1', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=10,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswl2', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=15,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswl3', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=30,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswl4', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=60,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswl5', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=90,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswl6', de.Rolling_Stats_withLookBack(columns= self.SELECTED_COLUMNS,window=120,lookback_divider=2,min_periods=None,verbose=True)),

            ('rswlc1', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=10,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswlc2', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=15,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswlc3', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=30,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswlc4', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=60,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswlc5', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=90,lookback_divider=2,min_periods=None,verbose=True)),
            ('rswlc6', de.Rolling_Stats_withLookBack_Compare(columns=self.SELECTED_COLUMNS,window=120,lookback_divider=2,min_periods=None,verbose=True))
                ])

        self.price_range_pipe = Pipeline([
            ('pltbc1', de.PriceLastTickBreachCount(column_pattern=[],columns=self.SELECTED_COLUMNS,last_ticks='10min',breach_type = self.dc.data.pltbc.breach_type)),
            ('pltbc2', de.PriceLastTickBreachCount(column_pattern=[],columns=self.SELECTED_COLUMNS,last_ticks='30min',breach_type = self.dc.data.pltbc.breach_type)),
            ('pltbc3', de.PriceLastTickBreachCount(column_pattern=[],columns=self.SELECTED_COLUMNS,last_ticks='60min',breach_type = self.dc.data.pltbc.breach_type)),            ('pltbc4', de.PriceLastTickBreachCount(column_pattern=[],columns=self.SELECTED_COLUMNS,last_ticks='120min',breach_type = self.dc.data.pltbc.breach_type)),

            ('pdr1', de.PreviousDaysRange(columns=self.SELECTED_COLUMNS,freq='d',shift=1,resample='1min',verbose=True)),
            ('pdr2', de.PreviousDaysRange(columns=self.SELECTED_COLUMNS,freq='d',shift=1,resample='15min',verbose=True)),
            ('pdr3', de.PreviousDaysRange(columns=self.SELECTED_COLUMNS,freq='d',shift=1,resample='30min',verbose=True)),
            ('pdr4', de.PreviousDaysRange(columns=self.SELECTED_COLUMNS,freq='w',shift=1,resample='1min',verbose=True)),

            ('gomc', de.GapOpenMinuteChart(columns=self.SELECTED_COLUMNS,verbose=True)),

            ('pdrhw2', de.PriceDayRangeHourWise(first_col = 'open',second_col='close',hour_range = self.dc.data.pdrhw.hour_range,range_type=self.dc.data.pdrhw.range_type)),
            ('pdrhw3', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = self.dc.data.pdrhw.hour_range,range_type=self.dc.data.pdrhw.range_type))
            ])

        self.label_creator_pipe = Pipeline([
            ('labelgenerator_1', de.LabelCreator_Light(freq='1min',shift=-15,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_2', de.LabelCreator_Light(freq='1min',shift=-30,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_3', de.LabelCreator_Light(freq='1min',shift=-60,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_4', de.LabelCreator_Light(freq='1min',shift=-45,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_5', de.LabelCreator_Super_Light(freq='1min',shift=-15,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_6', de.LabelCreator_Super_Light(freq='1min',shift=-30,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_7', de.LabelCreator_Super_Light(freq='1min',shift=-60,shift_column=self.dc.data.common.label_generator_col)),
            ('labelgenerator_8', de.LabelCreator_Super_Light(freq='1min',shift=-45,shift_column=self.dc.data.common.label_generator_col)),
            ])
