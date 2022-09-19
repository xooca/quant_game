import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import numpy as np
import data.data_utils as du
class BaseModel:
    def __init__(self,config):
        self.config = config
        self.trainer_option = self.config.model.trainer.generic.trainer_option
        self.define_parameters()

    def define_parameters(self):
        self.sampling_type = self.config.model.data.sampling_type
        self.using_print = self.config.model.model_metadata.print_option
        self.sampling_frac = self.config.model.data.sampling_frac
        self.target_column = self.config.model.data.target_column
        self.drop_columns = self.config.model.data.drop_columns
        self.label_columns = self.config.model.data.label_columns
        self.train_save_path = self.config.data.paths.train_save_path
        self.valid_save_path = self.config.data.paths.valid_save_path
        self.test_save_path = self.config.data.paths.test_save_path

    def sampling(self,df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        du.print_log(f"Sampling option is {self.sampling_type} ",self.using_print)
        if self.sampling_type == 'frac':
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(frac=self.sampling_frac))
        elif self.sampling_type == 'count':
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(self.sampling_frac))
        else:
            print("No sampling done")
        du.print_log(f"Shape of sample after sampling is {df.shape} ",self.using_print)
        return df

    def define_dataset(self,load_train = True,load_valid = True,load_test=False):
        self.valid = None
        self.train = None
        self.test = None
        self.drop_columns = list(self.drop_columns)
        self.label_columns = list(self.label_columns)
        self.label_columns = [col for col in self.label_columns if col != self.target_column]
        self.drop_columns = list(set(self.drop_columns + self.label_columns))
        if load_train:
            self.train = pd.read_csv(self.train_save_path)
            self.train = self.train[self.train[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.train.columns.tolist() if col not in self.drop_columns]
                #self.train = self.train.drop(self.drop_columns,axis=1)
                self.train = self.train[final_col]
            self.train = self.sampling(self.train)
            du.print_log(f"Train data creation is done",self.using_print)
            du.print_log(f"Train target value count is {self.train[self.target_column].value_counts()}",self.using_print)
        if load_valid:
            self.valid = pd.read_csv(self.valid_save_path)
            self.valid = self.valid[self.valid[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.valid.columns.tolist() if col not in self.drop_columns]
                #self.valid = self.valid.drop(self.drop_columns,axis=1)
                self.valid = self.valid[final_col]
            self.valid = self.sampling(self.valid)
            du.print_log(f"Valid data creation is done",self.using_print)
            du.print_log(f"Valid target value count is {self.valid[self.target_column].value_counts()}",self.using_print)
        if load_test:
            self.test = pd.read_csv(self.test_save_path)
            self.test = self.test[self.test[self.target_column]!='unknown']
            if len(self.drop_columns)>0:
                final_col = [col for col in self.test.columns.tolist() if col not in self.drop_columns]
                #self.test = self.test.drop(self.drop_columns,axis=1)
                self.test = self.test[final_col]
            self.test = self.sampling(self.test)
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


    