import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import numpy as np
import data.data_utils as du
from config.common import Config
        
class BaseModel:
    def __init__(self,config_obj):
        self.config_obj = config_obj
        self.model_metadata = {}
        self.define_parameters()

    def define_parameters(self):
        self.config_obj.initialize_all_config()

    def sampling(self,df):
        du.print_log(f"Sampling option is {self.sampling_type} ",self.using_print)
        if self.sampling_type == 'frac':
            #df.replace([np.inf, -np.inf], np.nan, inplace=True)
            #df.dropna(inplace=True)
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(frac=self.sampling_frac))

        elif self.sampling_type == 'count':
            #df.replace([np.inf, -np.inf], np.nan, inplace=True)
            #df.dropna(inplace=True)
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(self.sampling_frac))
        else:
            print("No sampling done")
        du.print_log(f"Shape of sample after sampling is {df.shape} ",self.using_print)
        return df

    def define_dataset(self):
        #self.drop_columns = list(self.drop_columns)
        #self.label_columns = list(self.label_columns)
        self.label_columns = [col for col in self.label_columns if col != self.target_column]
        self.drop_columns = list(set(self.drop_columns + self.label_columns))
        self.model_metadata.update({'label_columns':self.label_columns})
        self.model_metadata.update({'target_column':self.target_column})
        self.model_metadata.update({'drop_columns':self.drop_columns})
        self.model_metadata.update({'sampling_type':self.sampling_type})
        du.print_log(f"Length of drop columns is {len(self.drop_columns)}",self.using_print)
        if self.load_train:
            self.train = pd.read_csv(self.train_training_data_output_path,parse_dates=True,index_col='Unnamed: 0')
            self.model_metadata.update({'train_data_load_status':'success'})
            self.train = du.reduce_mem_usage(self.train)
            self.train = self.train[self.train[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.train.columns.tolist() if col not in self.drop_columns]
                #self.train = self.train.drop(self.drop_columns,axis=1)
                self.train = self.train[final_col]
            self.train = self.sampling(self.train)
            self.model_metadata.update({'training_data_columns':self.train.columns.tolist()})
            du.print_log(f"Train data creation is done",self.using_print)
            du.print_log(f"Train target value count is {self.train[self.target_column].value_counts()}",self.using_print)
        if self.load_valid:
            self.valid = pd.read_csv(self.train_validation_data_output_path,parse_dates=True,index_col='Unnamed: 0')
            self.model_metadata.update({'valid_data_load_status':'success'})
            self.valid = du.reduce_mem_usage(self.valid)
            self.valid = self.valid[self.valid[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.valid.columns.tolist() if col not in self.drop_columns]
                #self.valid = self.valid.drop(self.drop_columns,axis=1)
                self.valid = self.valid[final_col]
            self.valid = self.sampling(self.valid)
            self.model_metadata.update({'validation_data_columns':self.valid.columns.tolist()})
            du.print_log(f"Valid data creation is done",self.using_print)
            du.print_log(f"Valid target value count is {self.valid[self.target_column].value_counts()}",self.using_print)
        if self.load_test:
            self.test = pd.read_csv(self.train_testing_data_output_path,parse_dates=True,index_col='Unnamed: 0')
            self.model_metadata.update({'test_data_load_status':'success'})
            self.test = du.reduce_mem_usage(self.test)
            self.test = self.test[self.test[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.test.columns.tolist() if col not in self.drop_columns]
                #self.test = self.test.drop(self.drop_columns,axis=1)
                self.test = self.test[final_col]
            self.test = self.sampling(self.test)
            self.model_metadata.update({'test_data_columns':self.test.columns.tolist()})
            du.print_log(f"Test data creation is done",self.using_print)
            du.print_log(f"Test target value count is {self.test[self.target_column].value_counts()}",self.using_print)
        
    def model_saving(self):
        pass

    def model_tuning(self):
        pass

    def model_loading(self):
        pass

    def model_prediction(self):
        pass

    def model_evaluation(self):
        pass

    def trainer(self):
        pass


    