import pandas as pd
#import data_engine as de
from pycaret.classification import setup,compare_models,create_model,evaluate_model,plot_model,tune_model,blend_models,stack_models,finalize_model,save_model,load_model,predict_model
from pycaret.utils import check_metric
import numpy as np
from modelling.model_specs.base import BaseModel
import data.data_utils as du
import os
import omegaconf
from config.common.config import Config

class modelling(BaseModel):
    def __init__(self,master_config_path):
        BaseModel.__init__(self,master_config_path)
        print(f"Model spec is {self.model_spec_name}")
        self.define_parameters_custom()

    def define_parameters_custom(self):
        self.setup = self.config.model.trainer.setup
        self.tune_args = self.config.model.trainer.tune_model
        self.blend_args = self.config.model.trainer.blend_model
        self.stack_args = self.config.model.trainer.stack_model
        self.check_metric = self.config.model.trainer.check_metric
        self.create_model_args = self.config.model.trainer.create_model

    def initial_setup(self): 
        self.setup = dict(self.setup)
        self.setup['data'] = self.train
        if self.valid is not None:
            self.setup['test_data'] = self.valid
            du.print_log(f"Valid dataset is defined",self.using_print)
        self.experiment_setup = setup(**self.setup)
        self.model_metadata.update({'model_setup_args':self.setup})
        self.save_model_artifacts()
        
    def model_comparison(self, compare_option=1): 
        if self.previous_load_comparison and os.path.exists(self.model_comparison_save_path):
            du.print_log(f"Model comparison exists... Loading from from {self.model_comparison_save_path}",self.using_print)
            self.compared_model = du.load_object(object_path=self.model_comparison_save_path)
            du.print_log(f"Model comparison loaded from path {self.model_comparison_save_path}",self.using_print)
            self.model_metadata.update({'model_comparison_load_create_type':'loaded'})
        else:
            du.print_log(f"Model comparison doesnt exists at location {self.model_comparison_save_path}",self.using_print)
            self.compared_model = self.model_comparison_fn(compare_option=compare_option)
            du.print_log(f"Model comparison loaded ",self.using_print)
            du.save_object(object_path=self.model_comparison_save_path,obj=self.compared_model )
            du.print_log(f"Model comparison saved at location {self.model_comparison_save_path} ",self.using_print)
            self.model_metadata.update({'model_comparison_load_create_type':'created_saved'})
        self.model_metadata.update({'model_comparison_save_path':self.model_comparison_save_path})
        self.save_model_artifacts()

    def model_comparison_fn(self, compare_option=1):
        if compare_option == 1:
            arg_dict = dict(self.compare_models)
            compared_models = compare_models(**arg_dict) 
            self.model_metadata.update({'model_comparison_type':'created_with_custom_args'})
            self.model_metadata.update({'model_comparison_args':arg_dict})
        else:
            compared_models = compare_models()
            self.model_metadata.update({'model_comparison_type':'created_with_default_args'})
            self.model_metadata.update({'model_comparison_args':'default'})
        return compared_models[0]

    def model_creation(self): 
        if self.previous_load_creation and os.path.exists(self.model_creation_save_path):
            du.print_log(f"Model creation exists... Loading from from {self.model_creation_save_path}",self.using_print)
            self.created_model = du.load_object(object_path=self.model_creation_save_path)
            du.print_log(f"Model creation loaded from path {self.model_creation_save_path}",self.using_print)
            self.model_metadata.update({'model_creation_load_create_type':'loaded'})
        else:
            du.print_log(f"Model creation doesnt exists at location {self.model_creation_save_path}",self.using_print)
            self.created_model = self.model_creation_fn()
            du.print_log(f"Model creation loaded ",self.using_print)
            du.save_object(object_path=self.model_creation_save_path,obj=self.created_model )
            du.print_log(f"Model creation saved at location {self.model_creation_save_path} ",self.using_print)
            self.model_metadata.update({'model_creation_load_create_type':'created_saved'})
        self.model_metadata.update({'model_creation_save_path':self.model_creation_save_path})
        self.save_model_artifacts()

    def model_creation_fn(self):
        self.created_model = create_model(**self.create_model_args)
        self.model_metadata.update({'model_creation_args':self.create_model_args})

    def model_finalization(self):
        try:
            self.final_model = finalize_model(self.tuned_models)
            self.model_metadata.update({'model_finalization_status':'success'})
        except Exception as e:
            du.print_log(f"Unable to finalize model",self.using_print)
            du.print_log(f"Error encountered is {e}",self.using_print)
            self.model_metadata.update({'model_finalization_status':'failed'})
        self.save_model_artifacts()

    def model_saving(self):
        path = None
        try:
            path = f"{self.model_save_path}_{self.estimator}"
            save_model(self.final_model, path)
            self.model_metadata.update({'model_save_status':'success'})
        except Exception as e:
            du.print_log(f"Unable to save model to {path}",self.using_print)
            du.print_log(f"Error encountered is {e}",self.using_print)
            self.model_metadata.update({'model_save_status':'failed'})
        self.model_metadata.update({'model_path':path})
        self.save_model_artifacts()

    def model_tuning(self,from_compare=True): 
        if self.previous_load_tuned and os.path.exists(self.model_tuned_save_path):
            du.print_log(f"Tuned models exists... Loading from from {self.model_tuned_save_path}",self.using_print)
            self.tuned_model = du.load_object(object_path=self.model_tuned_save_path)
            du.print_log(f"Tuned models loaded from path {self.model_tuned_save_path}",self.using_print)
            self.model_metadata.update({'model_tuning_load_create_type':'loaded'})
        else:
            du.print_log(f"Tuned models doesnt exists at location {self.model_tuned_save_path}",self.using_print)
            self.tuned_model = self.model_tuning_fn(from_compare=from_compare)
            du.print_log(f"Tuned models loaded ",self.using_print)
            du.save_object(object_path=self.model_tuned_save_path,obj=self.tuned_models )
            du.print_log(f"Tuned models saved at location {self.model_tuned_save_path} ",self.using_print)
            self.model_metadata.update({'model_tuning_load_create_type':'created_saved'})
        self.model_metadata.update({'model_tuning_save_path':self.model_tuned_save_path})
        self.save_model_artifacts()

    def model_tuning_fn(self,from_compare=True):
        if from_compare:
            self.considered_model = self.compared_model
            self.model_metadata.update({'considered_model':'from_compare'})
            du.print_log(f"Model considered from 'from compare' ",self.using_print)
        else:
            self.considered_model = self.created_model
            self.model_metadata.update({'considered_model':'from_creation'})
            du.print_log(f"Model considered from 'from creation' ",self.using_print)
        try:
            arg_dict = dict(self.tune_args)
            arg_dict['estimator'] = self.considered_model
            tuned_model = tune_model(**arg_dict)
            du.print_log(f"Model tuning arguments are {arg_dict}",self.using_print)
            self.model_metadata.update({'model_tuning_status':'success'})
            self.model_metadata.update({'model_tuning_args':arg_dict})
            du.print_log(f"Model tuning success.... ",self.using_print)
        except Exception as e:
            du.print_log(f"Model tuning failed.... ",self.using_print)
            du.print_log(f"Error encountered is {e} ",self.using_print)
            self.model_metadata.update({'model_tuning_status':'failed'})
            tuned_model=None
        self.save_model_artifacts()
        return tuned_model

    def model_blending(self,estimator_list=None):
        arg_dict = dict(self.blend_args)
        try:
            if estimator_list is not None:
                arg_dict['estimator_list'] = estimator_list 
            else:
                arg_dict['estimator_list'] = self.tuned_model
                print(arg_dict)
            self.model_metadata.update({'model_blend_args':arg_dict})
            self.blender = blend_models(**arg_dict)
            self.model_metadata.update({'model_blend_status':'success'})
            self.model_metadata.update({'model_blend_no_of_models':len(arg_dict['estimator_list'])})
        except Exception as e:
            self.model_metadata.update({'model_blend_status':'failed'})
            self.model_metadata.update({'model_blend_no_of_models':0})
            du.print_log(f"Model blending failed.... ",self.using_print)
            du.print_log(f"Error encountered is {e} ",self.using_print)
        self.save_model_artifacts()

    def model_stacking(self,estimator_list=None,meta_model=None):
        arg_dict = dict(self.stack_args)
        try:
            if estimator_list is not None:
                arg_dict['estimator_list'] = estimator_list
                arg_dict['meta_model'] = meta_model
            else:
                arg_dict['estimator_list'] = self.tuned_models[1:]
                arg_dict['meta_model'] = self.tuned_models[0]
            self.stacker = stack_models(**arg_dict)
            self.model_metadata.update({'model_stack_args':arg_dict})
            self.model_metadata.update({'model_stack_status':'success'})
        except Exception as e:
            self.model_metadata.update({'model_stack_status':'failed'})
            du.print_log(f"Model stacking failed.... ",self.using_print)
            du.print_log(f"Error encountered is {e} ",self.using_print)
        self.save_model_artifacts()

    def predict_model(self,model,test,check_metric_flag=True):
        try:
            prediction = predict_model(model, data=test)
            pred_metric = None
            if check_metric_flag:
                check_metric_arg = dict(self.check_metric)
                check_metric_arg['actual'] = test[self.target_column]
                check_metric_arg['prediction'] = prediction['Label']
                pred_metric = check_metric(**check_metric_arg)
                self.model_metadata.update({'check_metric_arg':check_metric_arg})
            self.model_metadata.update({'model_predict_status':'success'})
            du.print_log(f"Model prediction success ",self.using_print)
        except Exception as e:
            self.model_metadata.update({'model_predict_status':'failed'})
            du.print_log(f"Model prediction failed.... ",self.using_print)
            du.print_log(f"Error encountered is {e} ",self.using_print)
        self.save_model_artifacts()        
        return prediction,pred_metric

    def model_loading(self):
        try:
            self.model_saved_path = du.load_object(object_path=self.model_metadata['model_path'])
            model = load_model(self.model_saved_path)
            self.model_metadata.update({'model_loading_status':'success'})
            self.model_metadata.update({'model_loaded_from_path':self.model_saved_path})
        except Exception as e:
            self.model_metadata.update({'model_loading_status':'failed'})
            du.print_log(f"Model Loading failed.... ",self.using_print)
            du.print_log(f"Error encountered is {e} ",self.using_print)
            model = None
        self.save_model_artifacts()
        return model          

    def model_prediction(self,test,check_metric_flag=False):
        model = self.model_loading()
        prediction,pred_metric = self.predict_model(model,test,check_metric_flag)
        self.model_metadata.update({'prediction':prediction})
        self.model_metadata.update({'pred_metric':pred_metric})
        self.save_model_artifacts()
        #du.save_object(object_path=self.model_prediction_save_path,obj=prediction_artifact)        

    def model_evaluation(self):
        model = self.model_loading()
        evaluation_artifact = evaluate_model(model)
        self.model_metadata.update({'evaluation_artifact':evaluation_artifact})
        self.save_model_artifacts()
        #du.save_object(object_path=self.model_evaluation_save_path,obj=evaluation_artifact)

    def save_model_artifacts(self):
        du.save_object(object_path=self.model_metadata_save_path,obj=self.model_metadata)
     
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
        self.model_finalization()
        du.print_log("******** Model Finalization Completed ************",self.using_print)
        self.model_saving()
        du.print_log("******** Model Saving Completed ************",self.using_print)
        self.model_evaluation()
        du.print_log("******** Model Evaluation Completed ************",self.using_print)
        self.model_prediction(self.test,check_metric_flag=True)
        du.print_log("******** Model Prediction Completed ************",self.using_print)
        #self.save_model_artifacts()
        #du.print_log("*************** Saved Metadata *****************",self.using_print)