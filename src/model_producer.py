# model_producer.py

import os
from config import config
from tqdm import tqdm  # <-- добавляем прогресс-бар

class Production:
    def __init__(self) -> None:
        self.input_dir = config['paths']['input_dir']
        self.temp_dir = config['paths']['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)

    def pipeline(self) -> None:
        """
        
        """
