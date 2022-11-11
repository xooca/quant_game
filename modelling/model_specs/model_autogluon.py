import autogluon.core as ag
from autogluon.tabular import TabularDataset
import numpy as np
class modelling:
    def __init__(self,config):
        self.config = config
        self.trainer_option = self.config.model.autogluon.trainer.generic.trainer_option

    def sampling(self,df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        if self.config.model.autogluon.model_metadata.sampling_type == 'frac':
            df = df.groupby(self.config.model.autogluon.model_metadata.target_column, group_keys=False).apply(lambda x: x.sample(frac=self.config.model.autogluon.model_metadata.sampling))
        elif self.config.model.autogluon.model_metadata.sampling_type == 'count':
            df = df.groupby(self.config.model.autogluon.model_metadata.target_column, group_keys=False).apply(lambda x: x.sample(self.config.model.autogluon.model_metadata.sampling))
        else:
            print("No sampling done")
        return df

    def define_dataset(self,load_train = True,load_valid = True,load_test=False):
        from autogluon.tabular import TabularDataset
        if load_train:
            self.train = TabularDataset(self.config.data.paths.train_save_path)
            self.train = self.train[self.train[self.config.model.autogluon.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.autogluon.model_metadata.drop_columns))>0:
                self.train = self.train.drop(list(self.config.model.autogluon.model_metadata.drop_columns),axis=1)
            self.train = self.sampling(self.train)
        if load_valid:
            self.valid = TabularDataset(self.config.data.paths.valid_save_path)
            self.valid = self.valid[self.valid[self.config.model.autogluon.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.autogluon.model_metadata.drop_columns))>0:
                self.valid = self.valid.drop(list(self.config.model.autogluon.model_metadata.drop_columns),axis=1)
            self.valid = self.sampling(self.valid)
        if load_test:
            self.test = TabularDataset(self.config.data.paths.test_save_path)
            self.test = self.test[self.test[self.config.model.autogluon.model_metadata.target_column]!='unknown']
            if len(list(self.config.model.autogluon.model_metadata.drop_columns))>0:
                self.test = self.test.drop(list(self.config.model.autogluon.model_metadata.drop_columns),axis=1)
            self.test = self.sampling(self.test)

    def define_hyperparameters(self):
        import autogluon.core as ag
        if self.config.model.autogluon.optimization.generic.use_default_parameter == 1:
            self.hyperparameters = 'default'
        if self.config.model.autogluon.optimization.generic.use_default_parameter == 0:
            self.hyperparameters = None
        else:
            self.hyperparameters = {}
            for i,j in self.config.model.autogluon.optimization.hyperparamters.items():
                self.hyperparameters[i] = {}
                for a,b in j.items():
                    print(isinstance(b,int))
                    if isinstance(b,int):
                        self.hyperparameters[i].update({a:b})
                    else:
                        l = eval(b)
                        self.hyperparameters[i].update({a:l})
        self.hyperparameter_tune_kwargs =self.config.model.autogluon.optimization.hyperparameter_tune_kwargs
        self.hyperparameter_tune_kwargs = dict(self.hyperparameter_tune_kwargs)

    def trainer(self):
        from autogluon.tabular import TabularPredictor
        import autogluon.core as ag
        if self.trainer_option == 1:
            self.predictor = TabularPredictor(label=self.config.model.autogluon.model_metadata.target_column, path=self.config.model.autogluon.model_metadata.model_save_path, eval_metric=self.config.model.autogluon.evaluation.eval_metric).fit(
            self.train, tuning_data=self.valid, time_limit=self.config.model.autogluon.trainer.generic.time_limit,
            hyperparameters=self.hyperparameters, hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,presets=self.config.model.autogluon.trainer.generic.presets)
        elif self.trainer_option == 2:
            self.predictor = TabularPredictor(label=self.config.model.autogluon.model_metadata.target_column, path=self.config.model.autogluon.model_metadata.model_save_path, eval_metric=self.config.model.autogluon.evaluation.eval_metric).fit(self.train,presets=self.config.model.autogluon.trainer.generic.presets)
        elif self.trainer_option == 3:
            self.predictor = TabularPredictor(label=self.config.model.autogluon.model_metadata.target_column, path=self.config.model.autogluon.model_metadata.model_save_path, eval_metric=self.config.model.autogluon.evaluation.eval_metric).fit(self.train,presets=self.config.model.autogluon.trainer.generic.presets,hyperparameters = dict(self.config.model.autogluon.trainer.hyperparameters))
        else:
            print(f"Option {self.trainer_option} is not supported")
        self.results = self.predictor.fit_summary(show_plot=True)

    def evaluate(self,test_df=None):
        from autogluon.tabular import TabularPredictor
        if self.config.model.autogluon.evaluation.load_predictor_from_path == 0:
            self.predictor = TabularPredictor.load(self.config.model.autogluon.model_metadata.model_save_path)
        if test_df is None:
            self.model_evaluation = self.predictor.evaluate(self.test, auxiliary_metrics=True)
        else:
            self.model_evaluation = self.predictor.evaluate(test_df, auxiliary_metrics=True)

    def predict(self,predict_df):
        from autogluon.tabular import TabularPredictor
        if self.config.model.autogluon.prediction.load_predictor_from_path == 0:
            self.predictor = TabularPredictor.load(self.config.model.autogluon.model_metadata.model_save_path)
        self.predictions = self.predictor.predict(predict_df)
        self.probabilities = self.predictor.predict_proba(predict_df)

    def save_artifacts(self):
        pass



    