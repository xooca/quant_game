#from operator import index
import data_engine as de
import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer,CategoricalImputer

data_save_path = './data/'
root_path = r'C:\\Users\\8prab\\Documents\\NSE_Equity_Futures_iEOD\\'
pattern = '*BANKNIFTY.txt'
data_name = 'banknifty'
#base_df= de.create_dataset(root_path,pattern,data_save_path,data_name)
base_df = pd.read_csv(r'C:\Users\8prab\Google Drive\Work\trading\data\banknifty\banknifty.csv')
column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap']
selected_cols = [m for m in base_df.columns.tolist() for mt in column_pattern if mt in m]

from sklearn.pipeline import Pipeline
pipe1 = Pipeline([
    ('nd', de.NormalizeDataset(columns = ['close','open','high','low'],impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
  #  ('labelgenerator_1', de.LabelCreator(freq='1min',shift=-15,shift_column='close')),
  #  ('labelgenerator_2', de.LabelCreator(freq='1min',shift=-30,shift_column='close')),
    ('tech_indicator', de.TechnicalIndicator(method_type = ['momentum','volatile','transform','pattern','overlap'])),
    ('nd2', de.NormalizeDataset(column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap'],fillna=True,fillna_method='bfill')),
    ('nd3', de.NormalizeDataset(column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap'],drop_na_rows=False,fillna=True,fillna_method='ffill')),
    ('fd1', de.FilterData(start_date='2021-01-01',end_date=None)),
    
    #('nd3', de.NormalizeDataset(columns = ['close','open','high','low'],fillna=True)),
    ])

pipe2 = Pipeline([
    ('rv1', de.RollingValues(columns=selected_cols,column_pattern=[],last_ticks=['10min','30min'],aggs=['mean','mean'],oper = ['-','='],verbose=True)),
    ('rv2', de.RollingValues(columns=selected_cols,column_pattern=[],last_ticks=['10min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('rv3', de.RollingValues(columns=selected_cols,column_pattern=[],last_ticks=['5min','30min'],aggs=['max','max'],oper = ['-','='],verbose=True)),
    ('ltgvc', de.LastTicksGreaterValuesCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks=10)),
    ('ltgvc1', de.LastTicksGreaterValuesCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks=30)),  
    ])

pipe3 = Pipeline([
    ('pdrhw', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = [('09:00', '10:30'),('10:30', '11:30')],range_type=['price_range','price_deviation_max_first_col'])),
    ('pv', de.PriceVelocity(freq='10min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=True)),
    ('pv2', de.PriceVelocity(freq='5min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=True)),
    ('ppi', de.PricePerIncrement(freq='D',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=True)),
    ])

pipe4 = Pipeline([
    ('pltbc', de.PriceLastTickBreachCount(column_pattern=[],columns=selected_cols,last_ticks='10min',breach_type = ['morethan','max'])),
    ('pltbc2', de.PriceLastTickBreachCount(column_pattern=[],columns=selected_cols,last_ticks='60min',breach_type = ['morethan','max'])),
    ('pdrhw2', de.PriceDayRangeHourWise(first_col = 'open',second_col='close',hour_range = [('09:00', '10:30'),('10:30', '11:30')],range_type=['price_range','price_deviation_max_first_col'])),
    ('pv3', de.PriceVelocity(freq='D',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=True)),
    ('ppi2', de.PricePerIncrement(freq='10min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=True)),
    ])

pipe5 = Pipeline([
    ('labelgenerator_1', de.LabelCreator(freq='1min',shift=-15,shift_column='close')),
    ('labelgenerator_2', de.LabelCreator(freq='1min',shift=-30,shift_column='close')),
    ])

def run_pipeline(pipe_list,df,pipeinfo_loc,data_loc,load_previous = True):
    import pickle
    import logging
    pipe_list_save = [col for col in pipe_list]
    if load_previous:
        try:
            with open(pipeinfo_loc, 'rb') as handle:
                pipe_list = pickle.load(handle)
            logging.info(f"Previous pipeline loaded from location {pipeinfo_loc}. Length of pipeline is {len(pipe_list)}")
            df = pd.read_csv(data_loc)
            logging.info(f"Previous data loaded from location {data_loc}. Shape of the data is {df.shape}")
        except Exception as e1:
            logging.info(f"File {pipeinfo_loc} is not loaded because of error : {e1}")
    for i, pipe in enumerate(pipe_list,1):
        logging.info('#'*100)
        logging.info(f"Pipeline {i} started. Shape of the data is {df.shape}")
        logging.info(pipe)
        df = pipe.fit(df).transform(df)
        pipe_list_save.remove(pipe)
        df.to_csv(data_loc)
        with open(pipeinfo_loc, 'wb') as handle:
            pickle.dump(pipe_list_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Pipeline {i} completed. Shape of the data is {df.shape}")

pipe_list =[pipe1,pipe2,pipe3,pipe4,pipe5]
pipeinfo_loc = r"E:\\\data\\trading\\pipe.pkl"
data_loc = r"E:\\\data\\trading\\base.csv"
run_pipeline(pipe_list,base_df,pipeinfo_loc,data_loc,load_previous=False)
