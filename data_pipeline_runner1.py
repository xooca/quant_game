import data.data_utils as du

master_config = du.initialize_config(overrides=["+master=master_config"],version_base=None, config_path="../config/banknifty/")
pipeline_obj = du.execute_data_pipeline(master_config.master.data)
pipeline_obj.execute_pipeline()
pipeline_obj.merge_pipeline()