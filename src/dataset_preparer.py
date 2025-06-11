# dataset_preparer.py

import os
from typing import List, Tuple
from .config import config
from tqdm import tqdm  # <-- добавляем прогресс-бар

class DatasetPreparation:
    def __init__(self) -> None:
        self.input_dir = config['paths']['input_dir']
        self.temp_dir = config['paths']['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)

    def pipeline(self, file_list: List[str]) -> Tuple:
        """
        загрузка датасета из файлов по списку file_list
        сбор DF
        получение дополнительных параметров:
        	типа скорости и ускорения изменения основных характеристик,
        	получение карт R-пиков и т.д.
        label-классификация, перевод в ohe
        нормирование/стандартизирование
        разделение X, y на выборки x_train, y_train, x_val, y_val, x_test, y_test        
        """
        x_train, y_train, x_val, y_val, x_test, y_test = '', '', '', '', '', ''
        return (x_train, y_train, x_val, y_val, x_test, y_test)
