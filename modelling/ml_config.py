import autogluon.core as ag

autogluon = {
    "evaluation": {"eval_metric":"accuracy"},
    "model" : {"model_save_path":''},
    "optimization" : {
            "generic"       :   {   
                                    "time_limit" : 2*60,  # train various models for ~2 min
                                    "num_trials" : 5,  # try at most 5 different hyperparameter configurations for each type of model
                                    "search_strategy" : 'auto',
                                    
                                },

            "nnoption"      :   {  # specifies non-default hyperparameter values for neural network models
                                    'num_epochs': 10,  # number of training epochs (controls training time of NN models)
                                    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
                                    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
                                    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
                                },

            "gbm_options"   :   {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
                                    'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
                                    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
                                },
            "hyperparameters" : {  # hyperparameters of each model type
                                    'GBM': gbm_options,
                                    'NN_TORCH': mlc.autogluon.optimization.nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                                 }, # When these keys are missing from hyperparameters dict, no models of that type are trained

            "hyperparameter_tune_kwargs" : {
                                    'num_trials': 'mlc.autogluon.generic.num_trials',
                                    'scheduler' : 'local',
                                    'searcher': 'mlc.autogluon.generic.search_strategy'
            
                                           }
                    },
    "date" : {'training_data_path':'',
              'validation_data_path':'',
              'test_data_path':'',
              'label':''}
}
