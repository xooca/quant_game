
import data.data_utils as du
from hydra.core.global_hydra import GlobalHydra
import hydra
import importlib
import pandas as pd

class execute_data_pipeline:
    def __init__(self,master_config):
        master_config = dict(master_config['master']['model'])
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.model_config = du.initialize_config(**master_config)
        self.model_spec_name = self.model_config.model.model_metadata.model_spec
        du.print_log(f"Model spec file is {self.model_spec_name}",self.using_print)
        self.model_spec = importlib.import_module(f"{self.model_spec_name}")

    def trainer(self):
        self.model_spec_obj = self.model_spec.modelling(self.model_config)
        self.model_spec_obj.define_dataset(load_train = True,load_valid = False,load_test=True)