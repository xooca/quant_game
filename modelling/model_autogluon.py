class modelling:
    def __init__(self,config,df):
        self.config = config

    def optimize(self,df):

    def train(self,df):
        import autogluon.core as ag
        from autogluon.tabular import TabularDataset, TabularPredictor
        train_data = TabularDataset(self.config.autogluon.optimization.training_data_path)
        val_data = TabularDataset(self.config.autogluon.optimization.validation_data_path)

    def evaluate(self,df):

    def predict(self,df):


    