import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
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
        
    def model_comparison(self, compare_option=1):
        if compare_option == 1:
            arg_dict = dict(self.config.model.pycaret.trainer.compare_models)
            self.model_compare = compare_models(**arg_dict) 
        else:
            self.model_compare = compare_models() 

    def model_creation_multiple(self):
        self.all_models =[] 
        for args,vals in self.config.model.pycaret.trainer.create_model.items():
            vals = dict(vals)
            model = create_model(**vals) 
            self.all_models.append(model)

    def model_creation(self):
        vals = self.config.model.pycaret.trainer.create_model
        self.all_models = [create_model(**vals)]

    def evaluate_tuned_model(self):
        if len(self.all_tuned_models) == 1:
            evaluate_model(self.all_tuned_models[0])
        else:
            for model in self.all_tuned_models:
                evaluate_model(model)

    def finalize_tune_model(self):
        if len(self.all_tuned_models) == 1:
            self.final_model = [finalize_model(self.all_tuned_models[0])]
        else:
            self.final_model = []
            for model in self.all_tuned_models:
                self.final_model = finalize_model(model)

    def save_models(self):
        if len(self.final_model) == 1:
            path = f"{self.config.model.pycaret.model_metadata.model_save_path}_{self.config.model.pycaret.trainer.create_model.estimator}"
            save_model(self.final_model[0], path)
            print(f"Model is save as path {path}")
        else:
            #for i,model in enumerate(self.final_model):
            if len(self.model_considered) == len(self.config.model.pycaret.trainer.create_model):
                for model,vals in zip(self.final_model,self.config.model.pycaret.trainer.create_model.items()):
                    arg_dict = dict(vals[-1])
                    path = f"{self.config.model.pycaret.model_metadata.model_save_path}_{arg_dict['estimator']}"
                    save_model(model, path)
                    print(f"Model is save as path {path}")
            else:
                for i,model in enumerate(self.final_model):
                    arg_dict = dict(vals[-1])
                    path = f"{self.config.model.pycaret.model_metadata.model_save_path}_{i}"
                    save_model(model, path)
                    print(f"Model is save as path {path}")

    def model_tuning(self,from_compare=True):
        self.all_tuned_models =[]
        if from_compare:
            self.model_considered = self.model_compare
        else:
            self.model_considered = self.all_models
        if len(self.model_considered) == len(self.config.model.pycaret.trainer.tune_model):
            for model,vals in zip(self.model_considered,self.config.model.pycaret.trainer.tune_model.items()):
                arg_dict = dict(vals[-1])
                arg_dict['estimator'] = model
                print(arg_dict)
                tuned_model = tune_model(**arg_dict) 
                self.all_tuned_models.append(tuned_model)
        else:
            for model in self.model_considered:
                arg_dict = dict(self.config.model.pycaret.trainer.tune_model_all)
                arg_dict['estimator'] = model
                print(arg_dict)
                tuned_model = tune_model(**arg_dict) 
                self.all_tuned_models.append(tuned_model)

    def model_blending(self,estimator_list=None):
        arg_dict = dict(self.config.model.pycaret.trainer.blend_model)
        if estimator_list is not None:
            arg_dict['estimator_list'] = estimator_list
            print(arg_dict)
        else:
            arg_dict['estimator_list'] = self.all_tuned_models
            print(arg_dict)
        self.blender = blend_models(**arg_dict)

    def model_stacking(self,estimator_list=None,meta_model=None):
        arg_dict = dict(self.config.model.pycaret.trainer.stack_model)
        if estimator_list is not None:
            arg_dict['estimator_list'] = estimator_list
            arg_dict['meta_model'] = meta_model
        else:
            arg_dict['estimator_list'] = self.all_tuned_models[1:]
            arg_dict['meta_model'] = self.all_tuned_models[0]
        self.stacker = stack_models(**arg_dict)

    def load_model_and_predict(self,test,check_metric_flag=True):
        path = f"{self.config.model.pycaret.model_metadata.model_save_path}_{self.config.model.pycaret.trainer.create_model.estimator}"
        #if self.config.model.pycaret.model_metadata.target_column in test.columns.tolist():
        #    test = test.drop(self.config.model.pycaret.model_metadata.target_column,axis=1)
        model = load_model(path)
        print(f"Model is save as path {path}")
        self.prediction = predict_model(model, data=test)
        if check_metric_flag:
            check_metric_arg = dict(self.config.model.pycaret.trainer.check_metric)
            check_metric_arg['actual'] = test[self.config.model.pycaret.model_metadata.target_column]
            check_metric_arg['prediction'] = self.prediction['Label']
            self.pred_metric = check_metric(**check_metric_arg)

    def load_and_predict_v1(self,test,check_metric_flag=True):
        if len(self.final_model) == 1:
            path = f"{self.config.model.pycaret.model_metadata.model_save_path}"
            #if self.config.model.pycaret.model_metadata.target_column in test.columns.tolist():
            #    test = test.drop(self.config.model.pycaret.model_metadata.target_column,axis=1)
            model = load_model(path)
            print(f"Model is save as path {path}")
            self.prediction = predict_model(model, data=test)
            if check_metric_flag:
                check_metric_arg = dict(self.config.model.pycaret.trainer.check_metric)
                check_metric_arg['actual'] = test[self.config.model.pycaret.model_metadata.target_column]
                check_metric_arg['prediction'] = self.prediction['Label']
                self.pred_metric = check_metric(**check_metric_arg)

        else:
            self.prediction = []
            self.pred_metric =[]
            for i,model in enumerate(self.final_model):
                path = f"{self.config.model.pycaret.model_metadata.model_save_path}_{i}"
                #if self.config.model.pycaret.model_metadata.target_column in test.columns.tolist():
                #    test = test.drop(self.config.model.pycaret.model_metadata.target_column,axis=1)
                model = load_model(path)
                self.prediction.append(predict_model(model, data=test))
                if check_metric_flag:
                    check_metric_arg = dict(self.config.model.pycaret.trainer.check_metric)
                    check_metric_arg['actual'] = test[self.config.model.pycaret.model_metadata.target_column]
                    check_metric_arg['prediction'] = self.prediction['Label']
                    self.pred_metric.append(check_metric(**check_metric_arg))

    def trainer_1(self):
        self.initial_setup()
        print("******** Setup Completed ************")
        self.model_comparison()
        print("******** Model Comparison Completed ************")
        self.model_tuning()
        print("******** Model Tuning Completed ************")
        self.model_blending()
        print("******** Model Blending Completed ************")


    def trainer(self):
        self.initial_setup()
        print("******** Setup Completed ************")
        self.model_creation( )
        print("******** Model Creation Completed ************")
        self.model_tuning(from_compare=False)
        print("******** Model Tuning Completed ************")
        self.finalize_tune_model()
        print("******** Model Finalization Completed ************")
        self.save_models()
        print("******** Model Saving Completed ************")
        self.load_and_predict(self,self.test,check_metric=True)


    