from sklearn.pipeline import Pipeline
import pickle
import logging
#import data.data_config as dc
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
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

warnings.filterwarnings('ignore')

os.getcwd()
pd.options.display.max_columns = None

def run_pipeline(pipe_list,df,pipeinfo_loc,data_loc,pipeline_save_loc,load_previous = True):
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
        with open(pipeline_save_loc, 'wb') as handle:
            pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Pipeline {i} completed. Shape of the data is {df.shape}")
    return df

def initialize_config(overrides,version_base=None, config_path="../config"):
    initialize(version_base=version_base, config_path=config_path)
    dc=compose(overrides= overrides)
    return dc

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
