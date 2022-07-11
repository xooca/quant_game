class modelling:
    def __init__(self,config):
        self.config = config

    def define_dataset(self,load_test=False):
        from autogluon.tabular import TabularDataset
        self.train = TabularDataset(self.config.data.path.train_save_path)
        self.valid = TabularDataset(self.config.data.path.valid_save_path)
        if load_test:
            self.test = TabularDataset(self.config.data.path.test_save_path)

    def define_hyperparameters(self):
        if self.config.model.autogluon.optimization.generic.use_default_parameter == 0:
            self.hyperparameters = 'default'
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
        if self.config.model.autogluon.optimization.generic.optimize == 0 or self.config.model.autogluon.optimization.generic.use_default_parameter==0:
            self.hyperparameter_tune_kwargs = None
        else:
            self.hyperparameter_tune_kwargs =self.config.model.autogluon.optimization.hyperparameter_tune_kwargs 

    def train(self):
        from autogluon.tabular import TabularPredictor
        import autogluon.core as ag
        self.predictor = TabularPredictor(label=self.config.model.autogluon.model_metadata.target_column, path=self.config.model.autogluon.model_metadata.model_save_path, eval_metric=self.config.model.autogluon.evaluation.eval_metric).fit(
        self.train, tuning_data=self.valid, time_limit=self.config.model.autogluon.optimization.generic.time_limit,
        hyperparameters=self.hyperparameters, hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs)
        self.results = self.predictor.fit_summary(show_plot=True)

    def evaluate(self,test_df=None):
        from autogluon.tabular import TabularPredictor
        if self.config.model.autogluon.evaluation.load_predictor_from_path == 0:
            self.predictor = TabularPredictor.load(self.config.model.autogluon.model_metadata.model_save_path)
        if test_df is None:
            self.model_evaluation = self.predictor.evaluate(self.test, auxiliary_metrics=False)
        else:
            self.model_evaluation = self.predictor.evaluate(test_df, auxiliary_metrics=False)

    def predict(self,predict_df):
        from autogluon.tabular import TabularPredictor
        if self.config.model.autogluon.prediction.load_predictor_from_path == 0:
            self.predictor = TabularPredictor.load(self.config.model.autogluon.model_metadata.model_save_path)
        self.predictions = self.predictor.predict(predict_df)
        self.probabilities = self.predictor.predict_proba(predict_df)



    