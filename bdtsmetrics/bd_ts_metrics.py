import argparse
import yaml
import os
import sys
import json
from .src.preprocess import load_my_own_data_csv, load_my_own_data_numpy, load_my_own_data_csv_v2
from .src.evaluation import evaluate_data
from .src.utils import write_json_data

class tsMetrics:
    """Implements time series metrics in python
      Usage:
      1. evaluate the quality of synthetic time series data

      :param config: metric config file.
      :type config: str
      :param real_data:  real time series data file
      :type real_data: str
      :param syn_data: synthetic time series data file
      :type syn_data: str
      :param seq_len: sequence length
      :type seq_len: int
    """

    def __init__(self, config, real_data, syn_data, seq_len):
      self.config = config
      self.real_data = real_data
      self.syn_data = syn_data
      self.seq_len = seq_len


    def load_config_from_file(self,config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)


    def evaluate(self):
        config = self.load_config_from_file(self.config)
        if self.real_data.endswith(".csv"):
            data = load_my_own_data_csv(self.real_data, self.seq_len)
        elif self.real_data.endswith(".npy"):
            data = load_my_own_data_numpy(self.real_data)
        else:
            raise Exception("Real data: wrong data format")

        if self.syn_data.endswith(".csv"):
            generated_data = load_my_own_data_csv(self.syn_data, self.seq_len)
        elif self.syn_data.endswith(".npy"):
            generated_data = load_my_own_data_numpy(self.syn_data)
        else:
            raise Exception("Synthetic data: wrong data format")

        if config['evaluation'].get('do_evaluation', True):
            dataset_name = config['preprocessing'].get('dataset_name', 'dataset')
            model_name = config['generation'].get('model', 'Betterdata-TS-Model')
            if 'dataset_name' not in config['evaluation']:
                config['evaluation']['evaluation'] = dataset_name
            if 'model' not in config['evaluation']:
                config['evaluation']['model'] = model_name
            results = evaluate_data(config['evaluation'], data, generated_data)

        if not os.path.isdir(os.path.join(os.getcwd(), 'result')):
            os.mkdir(os.path.join(os.getcwd(), 'result'))

        with open('./result/result.json', 'w') as f:
            json.dump(results, f)

        print('Program normal end.')




