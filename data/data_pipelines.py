
from sklearn.pipeline import Pipeline
import pickle
import logging
import data.data_engine as de
import pandas as pd
import numpy as np
#import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
import data.data_utils as du

dc = du.initialize_config(overrides=["+data=data"],version_base=None, config_path="../config/banknifty/")
OHLC_COLUMNS = list(dc.data.common.ohlc_column)
TECHNICAL_IND_PATTERN = list(dc.data.common.technical_indicator_col_pattern)
SELECTED_COLUMNS = list(dc.data.common.selected_columns)
TA_PIPE2_EXCLUDE = list(dc.data.common.ta_pipe2_exclude)

generic_pipeline = Pipeline([
                     ('nd_gen', de.NormalizeDataset(columns = OHLC_COLUMNS)),
                     ('read_fd', de.FilterData(start_date=dc.data.common.read_start_date,end_date=dc.data.common.read_end_date)),
                     ])

save_pipeline = Pipeline([
                     ('save_fd', de.FilterData(start_date=dc.data.common.save_start_date,end_date=dc.data.common.save_end_date)),
                     ])

technical_ind_pre_pipe = Pipeline([
    ('tech_ind_pre_ND', de.NormalizeDataset(columns = OHLC_COLUMNS,impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
    ])
    
technical_indicator_pipe1 = Pipeline([
    ('tech_indicator1', de.TechnicalIndicator(method_type = TECHNICAL_IND_PATTERN)),
    ('tech_indicator1_ND1', de.NormalizeDataset(column_pattern = OHLC_COLUMNS + TECHNICAL_IND_PATTERN,fillna=True,fillna_method='bfill')),
    ('tech_indicator1_ND2', de.NormalizeDataset(column_pattern = OHLC_COLUMNS + TECHNICAL_IND_PATTERN,fillna=True,fillna_method='ffill')),
    ])

technical_indicator_pipe2 = Pipeline([
    ('tech_indicator2', de.CreateTechnicalIndicatorUsingPandasTA(exclude=TA_PIPE2_EXCLUDE,verbose=True)),
   # ('tech_indicator2_ND1', de.NormalizeDataset(column_pattern = OHLC_COLUMNS,fillna=True,fillna_method='bfill')),
   # ('tech_indicator2_ND2', de.NormalizeDataset(column_pattern = OHLC_COLUMNS,drop_na_rows=False,fillna=True,fillna_method='ffill')),
    ])

technical_indicator_pipe3 = Pipeline([
    ('tech_indicator3', de.CreateTechnicalIndicatorUsingTA(volume_ta=False,verbose=True)),
  #  ('tech_indicator3_ND1', de.NormalizeDataset(column_pattern = OHLC_COLUMNS,fillna=True,fillna_method='bfill')),
  #  ('tech_indicator3_ND2', de.NormalizeDataset(column_pattern = OHLC_COLUMNS,drop_na_rows=False,fillna=True,fillna_method='ffill')),
    ])

rolling_values_pipe = Pipeline([
    ('rv1', de.RollingValues(columns= SELECTED_COLUMNS,column_pattern=[],last_ticks=['10min','30min'],aggs=['mean','mean'],oper = ['-','='],verbose=True)),
    ('rv2', de.RollingValues(columns= SELECTED_COLUMNS,column_pattern=[],last_ticks=['10min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('rv3', de.RollingValues(columns= SELECTED_COLUMNS,column_pattern=[],last_ticks=['5min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('rv4', de.RollingValues(columns= SELECTED_COLUMNS,column_pattern=[],last_ticks=['15min','60min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ])

last_tick_greater_values_pipe = Pipeline([    
    ('ltgvc1', de.LastTicksGreaterValuesCount(column_pattern=[],columns= SELECTED_COLUMNS,create_new_col = True,last_ticks=10)),
    ('ltgvc2', de.LastTicksGreaterValuesCount(column_pattern=[],columns= SELECTED_COLUMNS,create_new_col = True,last_ticks=15)),
    ('ltgvc3', de.LastTicksGreaterValuesCount(column_pattern=[],columns= SELECTED_COLUMNS,create_new_col = True,last_ticks=30)),  
    ])

zscore_log_percentage_chg_pipe = Pipeline([
   ('zs1', de.Zscoring(columns= SELECTED_COLUMNS,window=10,verbose=True)),
   ('zs2', de.Zscoring(columns= SELECTED_COLUMNS,window=15,verbose=True)),
   ('zs3', de.Zscoring(columns= SELECTED_COLUMNS,window=30,verbose=True)),
   ('zs4', de.Zscoring(columns= SELECTED_COLUMNS,window=60,verbose=True)),

   ('lgt', de.LogTransform(columns= SELECTED_COLUMNS,verbose=True)),

   ('pc1', de.PercentageChange(columns= SELECTED_COLUMNS,periods=10, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc2', de.PercentageChange(columns= SELECTED_COLUMNS,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc3', de.PercentageChange(columns= SELECTED_COLUMNS,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc4', de.PercentageChange(columns= SELECTED_COLUMNS,periods=60, fill_method='pad', limit=None, freq=None,verbose=True)),
    ])

percentile_transform_pipe = Pipeline([
   ('pt1', de.PercentileTransform(columns= SELECTED_COLUMNS,window=10,min_periods=None,quantile=0.75,verbose=True)),
   ('pt2', de.PercentileTransform(columns= SELECTED_COLUMNS,window=15,min_periods=None,quantile=0.75,verbose=True)),
   ('pt3', de.PercentileTransform(columns= SELECTED_COLUMNS,window=30,min_periods=None,quantile=0.75,verbose=True)),
   ('pt4', de.PercentileTransform(columns= SELECTED_COLUMNS,window=60,min_periods=None,quantile=0.75,verbose=True)),
    ])

rolling_rank_pipe = Pipeline([
   ('rr1', de.RollingRank(columns= SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
   ('rr2', de.RollingRank(columns= SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
   ('rr3', de.RollingRank(columns= SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
   ('rr4', de.RollingRank(columns= SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),
    ])

bin_transform_pipe = Pipeline([
   ('bt1', de.BinningTransform(columns= SELECTED_COLUMNS,window=10,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt2', de.BinningTransform(columns= SELECTED_COLUMNS,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt3', de.BinningTransform(columns= SELECTED_COLUMNS,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt4', de.BinningTransform(columns= SELECTED_COLUMNS,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
    ])

positive_negative_pipe = Pipeline([
   ('pnt1', de.PositiveNegativeTrends(columns= SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
   ('pnt2', de.PositiveNegativeTrends(columns= SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
   ('pnt3', de.PositiveNegativeTrends(columns= SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
   ('pnt4', de.PositiveNegativeTrends(columns= SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),
    ])

rolling_stats_pipe = Pipeline([
   ('rs1', de.Rolling_Stats(columns= SELECTED_COLUMNS,window=10,min_periods=None,verbose=True)),
   ('rs2', de.Rolling_Stats(columns= SELECTED_COLUMNS,window=15,min_periods=None,verbose=True)),
   ('rs3', de.Rolling_Stats(columns= SELECTED_COLUMNS,window=30,min_periods=None,verbose=True)),
   ('rs4', de.Rolling_Stats(columns= SELECTED_COLUMNS,window=60,min_periods=None,verbose=True)),

   ('rswl1', de.Rolling_Stats_withLookBack(columns= SELECTED_COLUMNS,window=10,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl2', de.Rolling_Stats_withLookBack(columns= SELECTED_COLUMNS,window=15,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl3', de.Rolling_Stats_withLookBack(columns= SELECTED_COLUMNS,window=30,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl4', de.Rolling_Stats_withLookBack(columns= SELECTED_COLUMNS,window=60,lookback_divider=2,min_periods=None,verbose=True)),

   ('rswlc1', de.Rolling_Stats_withLookBack_Compare(columns=SELECTED_COLUMNS,window=10,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc2', de.Rolling_Stats_withLookBack_Compare(columns=SELECTED_COLUMNS,window=15,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc3', de.Rolling_Stats_withLookBack_Compare(columns=SELECTED_COLUMNS,window=30,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc4', de.Rolling_Stats_withLookBack_Compare(columns=SELECTED_COLUMNS,window=60,lookback_divider=2,min_periods=None,verbose=True))
    ])

price_range_pipe = Pipeline([
    ('pltbc', de.PriceLastTickBreachCount(column_pattern=[],columns=SELECTED_COLUMNS,last_ticks='10min',breach_type = dc.data.pltbc.breach_type)),
    ('pltbc2', de.PriceLastTickBreachCount(column_pattern=[],columns=SELECTED_COLUMNS,last_ticks='60min',breach_type = dc.data.pltbc.breach_type)),

    ('pdr1', de.PreviousDaysRange(columns=SELECTED_COLUMNS,freq='d',shift=1,resample='1min',verbose=True)),
    ('pdr2', de.PreviousDaysRange(columns=SELECTED_COLUMNS,freq='w',shift=1,resample='1min',verbose=True)),

    ('gomc', de.GapOpenMinuteChart(columns=SELECTED_COLUMNS,verbose=True)),

    ('pdrhw2', de.PriceDayRangeHourWise(first_col = 'open',second_col='close',hour_range = dc.data.pdrhw.hour_range,range_type=dc.data.pdrhw.range_type)),
    ('pdrhw3', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = dc.data.pdrhw.hour_range,range_type=dc.data.pdrhw.range_type))
    ])

label_creator_pipe = Pipeline([
    ('labelgenerator_1', de.LabelCreator(freq='1min',shift=-15,shift_column=dc.data.common.label_generator_col)),
    ('labelgenerator_2', de.LabelCreator(freq='1min',shift=-30,shift_column=dc.data.common.label_generator_col)),
    ('labelgenerator_3', de.LabelCreator(freq='1min',shift=-60,shift_column=dc.data.common.label_generator_col)),
    ])