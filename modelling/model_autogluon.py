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
        self.hyperparameters = self.config.autogluon.autogluon.optimization.hyperparamters

    def optimize(self):
        from autogluon.tabular import TabularPredictor
        import autogluon.core as ag

        hyperparameter_tune_kwargs =self.config.autogluon.autogluon.optimization.hyperparameter_tune_kwargs
        self.predictor = TabularPredictor(label=self.config.autogluon.autogluon.model_metadata.target_column, eval_metric=self.config.autogluon.autogluon.evaluation.eval_metric).fit(
        self.train, tuning_data=self.valid, time_limit=self.config.autogluon.autogluon.optimization.generic.time_limit,
        hyperparameters=hyperparameters, hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)

    def train(self,df):
        import autogluon.core as ag
        from autogluon.tabular import TabularDataset, TabularPredictor
        self.train = TabularDataset(self.config.data.path.train_save_path)
        self.valid = TabularDataset(self.config.data.path.valid_save_path)

    def evaluate(self,df):

    def predict(self,df):


    