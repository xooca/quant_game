
import data.data_utils as du
import hydra
import importlib

class Trainer:
    def __init__(self,master_config):
        master_config = dict(master_config['master']['model'])
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.model_config = du.initialize_config(**master_config)
        self.model_spec_name = self.model_config.model.model_metadata.model_spec
        self.using_print = self.model_config.model.model_metadata.print_option
        du.print_log(f"Model spec file is {self.model_spec_name}",self.using_print)
        self.model_spec = importlib.import_module(f"{self.model_spec_name}")

    def train_model(self):
        self.model_spec_obj = self.model_spec.modelling(self.model_config)
        du.print_log(f"*******************************************",self.using_print)
        du.print_log(f"M O D E L    T R A I N I N G    S T A R T S",self.using_print)
        du.print_log(f"*******************************************",self.using_print)
        self.model_spec_obj.trainer()
        du.print_log(f"********************************************",self.using_print)
        du.print_log(f"M O D E L      T R A I N I N G       E N D S",self.using_print)
        du.print_log(f"********************************************",self.using_print)