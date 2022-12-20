
import data.data_utils as du
import hydra
import importlib
from config.common.config import Config


class Trainer:
    def __init__(self,master_config_path):
        self.master_config_path = master_config_path
        self.trainer_setup()

    def trainer_setup(self):
        self.base_config_obj = Config(self.master_config_path)
        self.base_config_obj.initialize_all_config()
        du.print_log(f"Model spec file is {self.base_config_obj.model_spec_name}",self.base_config_obj.using_print)
        self.model_spec = importlib.import_module(f"{self.base_config_obj.model_spec_name}")
        self.model_spec_obj = self.model_spec.modelling(self.master_config_path)

    def initialize_dataset(self):
        du.print_log(f"***************************************************",self.base_config_obj.using_print)
        du.print_log(f"D A T A S E T    D E F I N A T I O N    S T A R T S",self.base_config_obj.using_print)
        du.print_log(f"***************************************************",self.base_config_obj.using_print)
        self.model_spec_obj.define_dataset()
        self.model_spec_obj.initial_setup()
        du.print_log(f"***************************************************",self.base_config_obj.using_print)
        du.print_log(f"D A T A S E T      D E F I N A T I O N      E N D S",self.base_config_obj.using_print)
        du.print_log(f"***************************************************",self.base_config_obj.using_print)

    def train_model(self):
        du.print_log(f"*******************************************",self.base_config_obj.using_print)
        du.print_log(f"M O D E L    T R A I N I N G    S T A R T S",self.base_config_obj.using_print)
        du.print_log(f"*******************************************",self.base_config_obj.using_print)
        self.model_spec_obj.trainer()
        du.print_log(f"********************************************",self.base_config_obj.using_print)
        du.print_log(f"M O D E L      T R A I N I N G       E N D S",self.base_config_obj.using_print)
        du.print_log(f"********************************************",self.base_config_obj.using_print)