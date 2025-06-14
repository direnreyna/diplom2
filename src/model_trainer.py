# model_trainer.py

import os
from config import config
from tqdm import tqdm

class ModelTraining:
    def __init__(self, dataset_dict) -> None:
        self.dataset_dict = dataset_dict
        self.input_dir = config['paths']['input_dir']
        self.temp_dir = config['paths']['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)

    def pipeline(self) -> None:
        """
        
        """