import omegaconf
import data.data_utils as du
import hydra

class Config:
    def __init__(self,master_config_path):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        master_config = du.initialize_config(overrides=["+master=master_config"],version_base=None, config_path=master_config_path)
        master_config = dict(master_config['master']['model'])
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = du.initialize_config(**master_config)
        self.config_dict = dict(self.config)
        self.ind_config_dict = {}

    def extract(self,dict_in, dict_out):
        if isinstance(dict_in, omegaconf.dictconfig.DictConfig):
            dict_in = dict(dict_in)
        for key, value in dict_in.items():
            if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig): # If value itself is dictionary
                self.extract(value, dict_out)
            elif isinstance(value, str) or isinstance(value, list) or isinstance(value, int) or isinstance(value, omegaconf.listconfig.ListConfig) or isinstance(value, float):
                # Write to dict_out
                dict_out[key] = value
        return dict_out
    
    def initialize_all_config(self):
        t = self.extract(self.config_dict, self.ind_config_dict)
        for k,v in t.items():
          print(k,v)
          if isinstance(v, str):
            exec(f"self.{k}='{v}'")
          else:
            exec(f"self.{k}={v}")

class DefineConfig(Config):
    def __init__(self,master_config_path):
        Config.__init__(self,master_config_path)
        self.initialize_all_config()
        
    def __call__(self):
        self.initialize_all_config()