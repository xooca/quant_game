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

from sklearn.pipeline import Pipeline
pipe1 = Pipeline([
    ('nd', de.NormalizeDataset(columns = ['close','open','high','low'],impute_values=True,impute_type = 'mean_median_imputer',convert_to_floats = True)),
    #('labelgenerator_1', de.LabelCreator(freq='1min',shift=-15,shift_column='close')),
    #('labelgenerator_2', de.LabelCreator(freq='1min',shift=-30,shift_column='close')),
    ('tech_indicator', de.TechnicalIndicator(method_type = ['momentum','volatile','transform','pattern','overlap'])),
    ('nd2', de.NormalizeDataset(column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap'],fillna=True,fillna_method='bfill')),
    ('nd3', de.NormalizeDataset(column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap'],drop_na_rows=False,fillna=True,fillna_method='ffill')),
    
    #('nd3', de.NormalizeDataset(columns = ['close','open','high','low'],fillna=True)),
    ])
base_df = pipe1.fit(base_df).transform(base_df)

column_pattern = ['close','open','high','low','momentum','volatile','transform','pattern','overlap']

selected_cols = [m for m in base_df.columns.tolist() for mt in column_pattern if mt in m]

pipe2 = Pipeline([
    ('ltgvc', de.LastTicksGreaterValuesCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks=10)),
    ('ltgvc1', de.LastTicksGreaterValuesCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks=30)),
    ('pdrhw', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = [('09:00', '10:30'),('10:30', '11:30')],range_type=['price_range','price_deviation_max_first_col'])),
    ('pv', de.PriceVelocity(freq='10min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=False)),
    ('pv2', de.PriceVelocity(freq='5min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=False)),
    ('ppi', de.PricePerIncrement(freq='D',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=False)),
    ])
base_df = pipe2.fit(base_df).transform(base_df)

pipe3 = Pipeline([
    ('pltbc', de.PriceLastTickBreachCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks='10min',breach_type = ['morethan','max'])),
    ('pltbc2', de.PriceLastTickBreachCount(column_pattern=[],columns=selected_cols,create_new_col = True,last_ticks='60min',breach_type = ['morethan','max'])),
    ('pdrhw2', de.PriceDayRangeHourWise(first_col = 'high',second_col='low',hour_range = [('09:00', '10:30'),('10:30', '11:30')],range_type=['price_range','price_deviation_max_first_col'])),
    ('pv3', de.PriceVelocity(freq='D',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=False)),
    ('ppi2', de.PricePerIncrement(freq='10min',shift=5,shift_column=selected_cols,shift_column_pattern=[],verbose=False)),
    ])
base_df = pipe3.fit(base_df).transform(base_df)