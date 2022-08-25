import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models
import numpy as np

class modelling:
    def __init__(self,config):
        self.config = config
        self.trainer_option = self.config.model.pycaret.trainer.generic.trainer_option

    def sampling(self,df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if self.config.model.pycaret.model_metadata.sampling_type == 'frac':
            df = df.groupby(self.config.model.pycaret.model_metadata.target_column, group_keys=False).apply(lambda x: x.sample(frac=self.config.model.pycaret.model_metadata.sampling))
        elif self.config.model.pycaret.model_metadata.sampling_type == 'count':
            df = df.groupby(self.config.model.pycaret.model_metadata.target_column, group_keys=False).apply(lambda x: x.sample(self.config.model.pycaret.model_metadata.sampling))
        else:
            print("No sampling done")
        return df

    def define_dataset(self,load_train = True,load_valid = True,load_test=False):
        if load_train:
            self.train = pd.read_csv(self.config.data.paths.train_save_path)
            self.train = self.train[self.train[self.config.model.pycaret.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.pycaret.model_metadata.drop_columns))>0:
                self.train = self.train.drop(list(self.config.model.pycaret.model_metadata.drop_columns),axis=1)
            self.train = self.sampling(self.train)
        if load_valid:
            self.valid = pd.read_csv(self.config.data.paths.valid_save_path)
            self.valid = self.valid[self.valid[self.config.model.pycaret.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.pycaret.model_metadata.drop_columns))>0:
                self.valid = self.valid.drop(list(self.config.model.pycaret.model_metadata.drop_columns),axis=1)
            self.valid = self.sampling(self.valid)
        if load_test:
            self.test = pd.read_csv(self.config.data.paths.test_save_path)
            self.test = self.test[self.test[self.config.model.pycaret.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.pycaret.model_metadata.drop_columns))>0:
                self.test = self.test.drop(list(self.config.model.pycaret.model_metadata.drop_columns),axis=1)
            self.test = self.sampling(self.test)

    def initial_setup(self): 
        self.experiment_setup = setup(data = self.train, **self.config.model.pycaret.trainer.setup)

    def compare_models(self,compare_option==1):
        if compare_option == 1:
            arg_dict = dict(self.config.model.pycaret.trainer.compare_models)
            self.model_compare = compare_models(**arg_dict) 
        else:
            self.model_compare = compare_models() 

    def create_models(self):
        self.all_models =[] 
        for args,vals in self.config.model.pycaret.trainer.create_models.items():
            vals = dict(vals)
            model = create_model(**vals) 
            self.all_models.append(model)

    def tune_models(self):
        self.all_tuned_models =[] 
        for model,vals in zip(self.all_models,self.config.model.pycaret.trainer.tune_model.items()):
            arg_dict = dict(vals[-1])
            arg_dict['estimator'] = model
            tuned_model = tune_model(**arg_dict) 
            self.all_tuned_models.append(tuned_model)

    def blend_model(self,estimator_list=None):
        arg_dict = dict(self.config.model.pycaret.trainer.blend_model)
        if estimator_list is not None:
            arg_dict['estimator_list'] = estimator_list
        else:
            arg_dict['estimator_list'] = self.all_tuned_models
        self.blender = blend_models(**arg_dict)

    def stack_model(self,estimator_list,meta_model):
        arg_dict = dict(self.config.model.pycaret.trainer.stack_model)
        arg_dict['estimator_list'] = estimator_list
        arg_dict['meta_model'] = meta_model
        self.stacker = stack_models(**arg_dict)

    def trainer(self):
        self.initial_setup()
        self.compare_models()
        self.tune_models()

    def evaluate(self,test_df=None):
        evaluate_model(rf)

    def predict(self,predict_df):
        from autogluon.tabular import TabularPredictor
        if self.config.model.autogluon.prediction.load_predictor_from_path == 0:
            self.predictor = TabularPredictor.load(self.config.model.autogluon.model_metadata.model_save_path)
        self.predictions = self.predictor.predict(predict_df)
        self.probabilities = self.predictor.predict_proba(predict_df)

    def save_artifacts(self):
        pass



    