
from sklearn.pipeline import Pipeline
import pickle
import logging
import data.data_engine as de
import pandas as pd
import numpy as np
import data.data_config as dc

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

generic_pipeline = Pipeline([
                     ('nd_gen', de.NormalizeDataset(columns = dc.ohlc_column)),
                     ('read_fd', de.FilterData(start_date=dc.read_start_date,end_date=dc.read_end_date)),
                     ])

save_pipeline = Pipeline([
                     ('save_fd', de.FilterData(start_date=dc.save_start_date,end_date=dc.save_end_date)),
                     ])

technical_ind_pre_pipe = Pipeline([
    ('tech_ind_pre_ND', de.NormalizeDataset(columns = dc.ohlc_column,impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
    ])
    
technical_indicator_pipe1 = Pipeline([
    ('tech_indicator1', de.TechnicalIndicator(method_type = dc.technical_indicator_col_pattern)),
    ('tech_indicator1_ND1', de.NormalizeDataset(column_pattern = dc.ohlc_column + dc.technical_indicator_col_pattern,fillna=True,fillna_method='bfill')),
    ('tech_indicator1_ND2', de.NormalizeDataset(column_pattern = dc.ohlc_column + dc.technical_indicator_col_pattern,fillna=True,fillna_method='ffill')),
    ])

technical_indicator_pipe2 = Pipeline([
    ('tech_indicator2', de.CreateTechnicalIndicatorUsingPandasTA(exclude=dc.ta_pipe2_exclude,verbose=True)),
    ('tech_indicator2_ND1', de.NormalizeDataset(column_pattern = dc.ohlc_column,fillna=True,fillna_method='bfill')),
    ('tech_indicator2_ND2', de.NormalizeDataset(column_pattern = dc.ohlc_column,drop_na_rows=False,fillna=True,fillna_method='ffill')),
    ])

technical_indicator_pipe3 = Pipeline([
    ('tech_indicator3', de.CreateTechnicalIndicatorUsingTA(volume_ta=False,verbose=True)),
    ('tech_indicator3_ND1', de.NormalizeDataset(column_pattern = dc.ohlc_column,fillna=True,fillna_method='bfill')),
    ('tech_indicator3_ND2', de.NormalizeDataset(column_pattern = dc.ohlc_column,drop_na_rows=False,fillna=True,fillna_method='ffill')),
    ])

rolling_values_pipe = Pipeline([
    ('rv1', de.RollingValues(columns= dc.selected_columns,column_pattern=[],last_ticks=['10min','30min'],aggs=['mean','mean'],oper = ['-','='],verbose=True)),
    ('rv2', de.RollingValues(columns= dc.selected_columns,column_pattern=[],last_ticks=['10min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('rv3', de.RollingValues(columns= dc.selected_columns,column_pattern=[],last_ticks=['5min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('rv4', de.RollingValues(columns= dc.selected_columns,column_pattern=[],last_ticks=['15min','60min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ])

last_tick_greater_values_pipe = Pipeline([    
    ('ltgvc1', de.LastTicksGreaterValuesCount(column_pattern=[],columns= dc.selected_columns,create_new_col = True,last_ticks=10)),
    ('ltgvc2', de.LastTicksGreaterValuesCount(column_pattern=[],columns= dc.selected_columns,create_new_col = True,last_ticks=15)),
    ('ltgvc3', de.LastTicksGreaterValuesCount(column_pattern=[],columns= dc.selected_columns,create_new_col = True,last_ticks=30)),  
    ])

zscore_log_percentage_chg_pipe = Pipeline([
   ('zs1', de.Zscoring(columns= dc.selected_columns,window=10,verbose=True)),
   ('zs2', de.Zscoring(columns= dc.selected_columns,window=15,verbose=True)),
   ('zs3', de.Zscoring(columns= dc.selected_columns,window=30,verbose=True)),
   ('zs4', de.Zscoring(columns= dc.selected_columns,window=60,verbose=True)),

   ('lgt', de.LogTransform(columns= dc.selected_columns,verbose=True)),

   ('pc1', de.PercentageChange(columns= dc.selected_columns,periods=10, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc2', de.PercentageChange(columns= dc.selected_columns,periods=15, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc3', de.PercentageChange(columns= dc.selected_columns,periods=30, fill_method='pad', limit=None, freq=None,verbose=True)),
   ('pc4', de.PercentageChange(columns= dc.selected_columns,periods=60, fill_method='pad', limit=None, freq=None,verbose=True)),
    ])

percentile_transform_pipe = Pipeline([
   ('pt1', de.PercentileTransform(columns= dc.selected_columns,window=10,min_periods=None,quantile=0.75,verbose=True)),
   ('pt2', de.PercentileTransform(columns= dc.selected_columns,window=15,min_periods=None,quantile=0.75,verbose=True)),
   ('pt3', de.PercentileTransform(columns= dc.selected_columns,window=30,min_periods=None,quantile=0.75,verbose=True)),
   ('pt4', de.PercentileTransform(columns= dc.selected_columns,window=60,min_periods=None,quantile=0.75,verbose=True)),
    ])

rolling_rank_pipe = Pipeline([
   ('rr1', de.RollingRank(columns= dc.selected_columns,window=10,min_periods=None,verbose=True)),
   ('rr2', de.RollingRank(columns= dc.selected_columns,window=15,min_periods=None,verbose=True)),
   ('rr3', de.RollingRank(columns= dc.selected_columns,window=30,min_periods=None,verbose=True)),
   ('rr4', de.RollingRank(columns= dc.selected_columns,window=60,min_periods=None,verbose=True)),
    ])

bin_transform_pipe = Pipeline([
   ('bt1', de.BinningTransform(columns= dc.selected_columns,window=10,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt2', de.BinningTransform(columns= dc.selected_columns,window=15,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt3', de.BinningTransform(columns= dc.selected_columns,window=30,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
   ('bt4', de.BinningTransform(columns= dc.selected_columns,window=60,min_period=None,get_current_row_bin=True,n_bins=5,verbose=True)),
    ])

positive_negative_pipe = Pipeline([
   ('pnt1', de.PositiveNegativeTrends(columns= dc.selected_columns,window=10,min_periods=None,verbose=True)),
   ('pnt2', de.PositiveNegativeTrends(columns= dc.selected_columns,window=15,min_periods=None,verbose=True)),
   ('pnt3', de.PositiveNegativeTrends(columns= dc.selected_columns,window=30,min_periods=None,verbose=True)),
   ('pnt4', de.PositiveNegativeTrends(columns= dc.selected_columns,window=60,min_periods=None,verbose=True)),
    ])

rolling_stats_pipe = Pipeline([
   ('rs1', de.Rolling_Stats(columns= dc.selected_columns,window=10,min_periods=None,verbose=True)),
   ('rs2', de.Rolling_Stats(columns= dc.selected_columns,window=15,min_periods=None,verbose=True)),
   ('rs3', de.Rolling_Stats(columns= dc.selected_columns,window=30,min_periods=None,verbose=True)),
   ('rs4', de.Rolling_Stats(columns= dc.selected_columns,window=60,min_periods=None,verbose=True)),

   ('rswl1', de.Rolling_Stats_withLookBack(columns= dc.selected_columns,window=10,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl2', de.Rolling_Stats_withLookBack(columns= dc.selected_columns,window=15,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl3', de.Rolling_Stats_withLookBack(columns= dc.selected_columns,window=30,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswl4', de.Rolling_Stats_withLookBack(columns= dc.selected_columns,window=60,lookback_divider=2,min_periods=None,verbose=True)),

   ('rswlc1', de.Rolling_Stats_withLookBack_Compare(columns=dc.selected_columns,window=10,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc2', de.Rolling_Stats_withLookBack_Compare(columns=dc.selected_columns,window=15,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc3', de.Rolling_Stats_withLookBack_Compare(columns=dc.selected_columns,window=30,lookback_divider=2,min_periods=None,verbose=True)),
   ('rswlc4', de.Rolling_Stats_withLookBack_Compare(columns=dc.selected_columns,window=60,lookback_divider=2,min_periods=None,verbose=True))
    ])

price_range_pipe = Pipeline([
    ('pltbc', de.PriceLastTickBreachCount(column_pattern=[],columns=dc.selected_columns,last_ticks='10min',breach_type = ['morethan','max'])),
    ('pltbc2', de.PriceLastTickBreachCount(column_pattern=[],columns=dc.selected_columns,last_ticks='60min',breach_type = ['morethan','max'])),

    ('pdr1', de.PreviousDaysRange(columns=dc.selected_columns,freq='d',shift=1,resample='1min',verbose=True)),
    ('pdr2', de.PreviousDaysRange(columns=dc.selected_columns,freq='w',shift=1,resample='1min',verbose=True)),

    ('gomc', de.GapOpenMinuteChart(columns=dc.selected_columns,verbose=True)),

    ('pdrhw2', de.PriceDayRangeHourWise(first_col = 'open',second_col='close',hour_range = [('09:00', '10:30'),('10:30', '11:30'),('11:30', '12:30'),('12:30', '01:30'),('02:30', '15:30')],range_type=['price_range','price_deviation_max_first_col','price_deviation_min_first_col','price_deviation_max_second_col','price_deviation_min_second_col'])),

    ('pdrhw3', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = [('09:00', '10:30'),('10:30', '11:30'),('11:30', '12:30'),('12:30', '01:30'),('02:30', '15:30')],range_type=['price_range','price_deviation_max_first_col','price_deviation_min_first_col','price_deviation_max_second_col','price_deviation_min_second_col']))

    ])

label_creator_pipe = Pipeline([
    ('labelgenerator_1', de.LabelCreator(freq='1min',shift=-15,shift_column=dc.label_generator_col)),
    ('labelgenerator_2', de.LabelCreator(freq='1min',shift=-30,shift_column=dc.label_generator_col)),
    ('labelgenerator_3', de.LabelCreator(freq='1min',shift=-60,shift_column=dc.label_generator_col)),
    ])
