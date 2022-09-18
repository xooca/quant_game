import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import numpy as np

class BaseModel:
    def __init__(self,config):
        self.config = config
        self.trainer_option = self.config.model.trainer.generic.trainer_option
        self.define_parameters()

    def define_parameters(self):
        self.sampling_type = self.config.model.data.sampling_type
        self.sampling_frac = self.config.model.data.sampling_frac
        self.target_column = self.config.model.data.target_column
        self.drop_columns = self.config.model.data.drop_columns
        self.train_save_path = self.config.data.paths.train_save_path
        self.valid_save_path = self.config.data.paths.valid_save_path
        self.test_save_path = self.config.data.paths.test_save_path

    def sampling(self,df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if self.sampling_type == 'frac':
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(frac=self.sampling_frac))
        elif self.sampling_type == 'count':
            df = df.groupby(self.target_column, group_keys=False).apply(lambda x: x.sample(self.sampling_frac))
        else:
            print("No sampling done")
        return df

    def define_dataset(self,load_train = True,load_valid = True,load_test=False):
        if load_train:
            self.train = pd.read_csv(self.train_save_path)
            self.train = self.train[self.train[self.target_column]!='unknown']
            if len(list(self.drop_columns))>0:
                self.train = self.train.drop(list(self.drop_columns),axis=1)
            self.train = self.sampling(self.train)
        if load_valid:
            self.valid = pd.read_csv(self.valid_save_path)
            self.valid = self.valid[self.valid[self.target_column]!='unknown']
            if len(list(self.drop_columns))>0:
                self.valid = self.valid.drop(list(self.drop_columns),axis=1)
            self.valid = self.sampling(self.valid)
        if load_test:
            self.test = pd.read_csv(self.test_save_path)
            self.test = self.test[self.test[self.target_column]!='unknown']
            if len(list(self.drop_columns))>0:
                self.test = self.test.drop(list(self.drop_columns),axis=1)
            self.test = self.sampling(self.test)

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


    