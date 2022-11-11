

from kiteconnect import KiteConnect
from selenium import webdriver
import time
import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf
from data.data_utils import initialize_config,print_log
import requests
import pandas as pd
from config.common.config import DefineConfig


class TradeApiData:
    def __init__(self,master_config_path):
        DefineConfig.__init__(self,master_config_path)

    def autenticate_api(self):
        pass

    def create_url(self,url=None):
        pass

    def get_data_from_api(self,url=None,read_action=None):
        pass

