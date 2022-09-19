import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import numpy as np
from modelling.base import BaseModel
import data.data_utils as du
import os
import omegaconf

class modelling(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)
        self.define_parameters_pycaret()

    def define_parameters_pycaret(self):
        self.compare_models = self.config.model.trainer.compare_models
        self.setup = self.config.model.trainer.setup
        self.create_model = self.config.model.trainer.create_model
        self.model_save_path = self.config.model.model_metadata.model_save_path
        self.model_evaluation_save_path = self.config.model.model_metadata.model_evaluation_save_path
        self.model_prediction_save_path = self.config.model.model_metadata.model_prediction_save_path
        self.blend_model = self.config.model.trainer.blend_model
        self.using_print = self.config.model.model_metadata.print_option
        
        self.model_setup_save_path = self.config.model.model_metadata.model_setup_save_path
        self.model_comparison_save_path = self.config.model.model_metadata.model_comparison_save_path
        self.model_creation_save_path = self.config.model.model_metadata.model_creation_save_path
        self.model_tuned_save_path = self.config.model.model_metadata.model_tuned_save_path

        self.previous_load_setup = self.config.model.model_metadata.previous_load_setup
        self.previous_load_comparison = self.config.model.model_metadata.previous_load_comparison
        self.previous_load_creation = self.config.model.model_metadata.previous_load_creation
        self.previous_load_tuned = self.config.model.model_metadata.previous_load_tuned

    def initial_setup(self): 
        self.setup = dict(self.setup)
        self.setup['data'] = self.train
        if self.test is not None:
            self.setup['test_data'] = self.test
        self.experiment_setup = setup(**self.setup)

    def initial_setup_v1(self): 
        if self.previous_load_setup:
            if os.path.exists(self.model_setup_save_path):
                du.print_log(f"Setup exists... Loading from from {self.model_setup_save_path}",self.using_print)
                self.experiment_setup = du.load_object(object_path=self.model_setup_save_path)
                du.print_log(f"Set up loaded from path {self.model_setup_save_path}",self.using_print)
            else:
                du.print_log(f"Set up doesnt exists at location {self.model_setup_save_path}",self.using_print)
                self.experiment_setup = setup(data = self.train, **self.setup)
                du.print_log(f"Set up loaded ",self.using_print)
                du.save_object(object_path=self.model_setup_save_path,obj=self.experiment_setup )
                du.print_log(f"Set up saved at location {self.model_setup_save_path} ",self.using_print)
        else:
            du.print_log(f"Loading setup",self.using_print)
            self.experiment_setup = setup(data = self.train, **self.setup)
            du.print_log(f"Set up loaded ",self.using_print)
            du.save_object(object_path=self.model_setup_save_path,obj=self.experiment_setup )
            du.print_log(f"Set up saved at location {self.model_setup_save_path} ",self.using_print)
        
    def model_comparison(self, compare_option=1): 
        if self.previous_load_comparison:
            if os.path.exists(self.model_comparison_save_path):
                du.print_log(f"Model comparison exists... Loading from from {self.model_comparison_save_path}",self.using_print)
                self.compared_models = du.load_object(object_path=self.model_comparison_save_path)
                du.print_log(f"Model comparison loaded from path {self.model_comparison_save_path}",self.using_print)
            else:
                du.print_log(f"Model comparison doesnt exists at location {self.model_comparison_save_path}",self.using_print)
                self.compared_models = self.model_comparison_fn(compare_option=compare_option)
                du.print_log(f"Model comparison loaded ",self.using_print)
                du.save_object(object_path=self.model_comparison_save_path,obj=self.compared_models )
                du.print_log(f"Model comparison saved at location {self.model_comparison_save_path} ",self.using_print)
        else:
            du.print_log(f"Loading Model comparison",self.using_print)
            self.compared_models = self.model_comparison_fn(compare_option=compare_option)
            du.print_log(f"Model comparison loaded ",self.using_print)
            du.save_object(object_path=self.model_comparison_save_path,obj=self.compared_models )
            du.print_log(f"Model comparison saved at location {self.model_comparison_save_path} ",self.using_print)

    def model_comparison_fn(self, compare_option=1):
        if compare_option == 1:
            arg_dict = dict(self.compare_models)
            compared_models = compare_models(**arg_dict) 
        else:
            compared_models = compare_models() 
        return compared_models

    def model_creation(self): 
        if self.previous_load_creation:
            if os.path.exists(self.model_creation_save_path):
                du.print_log(f"Model creation exists... Loading from from {self.model_creation_save_path}",self.using_print)
                self.created_models = du.load_object(object_path=self.model_creation_save_path)
                du.print_log(f"Model creation loaded from path {self.model_creation_save_path}",self.using_print)
            else:
                du.print_log(f"Model creation doesnt exists at location {self.model_creation_save_path}",self.using_print)
                self.created_models = self.model_creation_fn()
                du.print_log(f"Model creation loaded ",self.using_print)
                du.save_object(object_path=self.model_creation_save_path,obj=self.created_models )
                du.print_log(f"Model creation saved at location {self.model_creation_save_path} ",self.using_print)
        else:
            du.print_log(f"Loading Model creation",self.using_print)
            self.created_models = self.model_creation_fn()
            du.print_log(f"Model creation loaded ",self.using_print)
            du.save_object(object_path=self.model_creation_save_path,obj=self.created_models )
            du.print_log(f"Model creation saved at location {self.model_creation_save_path} ",self.using_print)

    def model_creation_fn(self):
        if not any(isinstance(j,omegaconf.dictconfig.DictConfig) for i,j in self.create_model.items()):
            created_models = [create_model(**self.create_model)]
        else:
            created_models =[] 
            for args,vals in self.config.model.trainer.create_model.items():
                vals = dict(vals)
                model = create_model(**vals) 
                created_models.append(model)
        return created_models

    def finalize_tuned_model(self):
        if len(self.tuned_models) == 1:
            self.final_model = [finalize_model(self.tuned_models[0])]
        else:
            self.final_model = []
            for model in self.tuned_models:
                self.final_model = finalize_model(model)

    def model_saving(self):
        model_saved_paths = []
        if len(self.final_model) == 1:
            path = f"{self.model_save_path}_{self.config.model.trainer.create_model.estimator}"
            save_model(self.final_model[0], path)
            print(f"Model is save as path {path}")
            model_saved_paths.append(path)
        else:
            if len(self.considered_models) == len(self.create_model):
                for model,vals in zip(self.final_model,self.create_model.items()):
                    arg_dict = dict(vals[-1])
                    path = f"{self.model_save_path}_{arg_dict['estimator']}"
                    save_model(model, path)
                    print(f"Model is save as path {path}")
                    model_saved_paths.append(path)
            else:
                for i,model in enumerate(self.final_model):
                    arg_dict = dict(vals[-1])
                    path = f"{self.model_save_path}_{i}"
                    save_model(model, path)
                    print(f"Model is save as path {path}")
                    model_saved_paths.append(path)
        du.save_object(object_path=f"{self.model_save_path}_type.pkl",obj=model_saved_paths)

    def model_tuning(self,from_compare=True): 
        if self.previous_load_creation:
            if os.path.exists(self.model_tuned_save_path):
                du.print_log(f"Tuned models exists... Loading from from {self.model_tuned_save_path}",self.using_print)
                self.tuned_models = du.load_object(object_path=self.model_tuned_save_path)
                du.print_log(f"Tuned models loaded from path {self.model_tuned_save_path}",self.using_print)
            else:
                du.print_log(f"Tuned models doesnt exists at location {self.model_tuned_save_path}",self.using_print)
                self.tuned_models = self.model_tuning_fn(from_compare=from_compare)
                du.print_log(f"Tuned models loaded ",self.using_print)
                du.save_object(object_path=self.model_tuned_save_path,obj=self.tuned_models )
                du.print_log(f"Tuned models saved at location {self.model_tuned_save_path} ",self.using_print)
        else:
            du.print_log(f"Loading tuned models",self.using_print)
            self.tuned_models = self.model_tuning_fn(from_compare=from_compare)
            du.print_log(f"Tuned models loaded ",self.using_print)
            du.save_object(object_path=self.model_tuned_save_path,obj=self.tuned_models )
            du.print_log(f"Tuned models saved at location {self.model_tuned_save_path} ",self.using_print)

    def model_tuning_fn(self,from_compare=True):
        tuned_models =[]
        if from_compare:
            self.considered_models = self.compared_models
        else:
            self.considered_models = self.created_models
        if len(self.considered_models) == len(self.config.model.trainer.tune_model):
            for model,vals in zip(self.considered_models,self.config.model.trainer.tune_model.items()):
                arg_dict = dict(vals[-1])
                arg_dict['estimator'] = model
                print(arg_dict)
                tuned_model = tune_model(**arg_dict) 
                tuned_models.append(tuned_model)
        else:
            for model in self.considered_models:
                arg_dict = dict(self.config.model.trainer.tune_model_all)
                arg_dict['estimator'] = model
                print(arg_dict)
                tuned_model = tune_model(**arg_dict) 
                tuned_models.append(tuned_model)
        return tuned_models

    def model_blending(self,estimator_list=None):
        arg_dict = dict(self.config.model.trainer.blend_model)
        if estimator_list is not None:
            arg_dict['estimator_list'] = estimator_list
            print(arg_dict)
        else:
            arg_dict['estimator_list'] = self.tuned_models
            print(arg_dict)
        self.blender = blend_models(**arg_dict)

    def model_stacking(self,estimator_list=None,meta_model=None):
        arg_dict = dict(self.config.model.trainer.stack_model)
        if estimator_list is not None:
            arg_dict['estimator_list'] = estimator_list
            arg_dict['meta_model'] = meta_model
        else:
            arg_dict['estimator_list'] = self.tuned_models[1:]
            arg_dict['meta_model'] = self.tuned_models[0]
        self.stacker = stack_models(**arg_dict)

    def predict_model(self,model,test,check_metric_flag=True):
        prediction = predict_model(model, data=test)
        pred_metric = None
        if check_metric_flag:
            check_metric_arg = dict(self.config.model.trainer.check_metric)
            check_metric_arg['actual'] = test[self.target_column]
            check_metric_arg['prediction'] = prediction['Label']
            pred_metric = check_metric(**check_metric_arg)
        return prediction,pred_metric

    def load_models(self):
        self.model_saved_paths = du.load_object(object_path=f"{self.model_save_path}_type.pkl")
        all_saved_models = []
        for model_path in self.model_saved_paths:
            model = load_model(model_path)
            all_saved_models.append(model)
        return all_saved_models

    def model_prediction(self,test,check_metric_flag=False):
        models = self.load_models()
        prediction_artifacts = []
        for model in models:
            prediction_artifact = {}
            prediction,pred_metric = self.predict_model(model,test,check_metric_flag)
            prediction_artifact.update({'prediction':prediction})
            prediction_artifact.update({'pred_metric':pred_metric})
            prediction_artifact.update({'model':model})
            prediction_artifacts.append(prediction_artifact)
        du.save_object(object_path=self.model_prediction_save_path,obj=prediction_artifacts)

    def model_evaluation(self):
        models = self.load_models()
        evaluation_artifacts =[]
        for model in models:
            evaluation_artifacts.append(evaluate_model(model))
        du.save_object(object_path=self.model_evaluation_save_path,obj=evaluation_artifacts)

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
        #self.model_spec_obj = self.model_spec.modelling(self.model_config)

        #self.initial_setup()
        #du.print_log("******** Setup Completed ************",self.using_print)
        self.model_creation()
        du.print_log("******** Model Creation Completed ************",self.using_print)
        self.model_tuning(from_compare=False)
        du.print_log("******** Model Tuning Completed ************",self.using_print)
        self.finalize_tuned_model()
        du.print_log("******** Model Finalization Completed ************",self.using_print)
        self.model_saving()
        du.print_log("******** Model Saving Completed ************",self.using_print)
        self.model_evaluation()
        du.print_log("******** Model Evaluation Completed ************",self.using_print)
        self.model_prediction(self.test,check_metric_flag=True)
        du.print_log("******** Model Prediction Completed ************",self.using_print)