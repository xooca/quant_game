from functools import partial
from datetime import datetime

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