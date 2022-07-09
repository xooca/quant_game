import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import ml_config as mlc

train_data = TabularDataset(mlc.autogluon.optimization.training_data_path)
val_data = TabularDataset(mlc.autogluon.optimization.validation_data_path)


hyperparameters = {  # hyperparameters of each model type
                   'GBM': mlc.autogluon.optimization.gbm_options,
                   'NN_TORCH': mlc.autogluon.optimization.nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': mlc.autogluon.generic.num_trials,
    'scheduler' : 'local',
    'searcher': mlc.autogluon.generic.search_strategy,
}

predictor = TabularPredictor(label=mlc.autogluon.data.label, eval_metric=mlc.autogluon.evalaution.eval_metric,path=mlc.autogluon.model.model_save_path).fit(
    train_data, tuning_data=val_data, time_limit=mlc.autogluon.generic.time_limit,
    hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)

