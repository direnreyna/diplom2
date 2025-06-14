# main.py

import os
import sys
# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

#import src.config
from config import config

from file_management import FileManagement
from dataset_preprocessing import DatasetPreprocessing
from model_trainer import ModelTraining
from model_producer import Production

dataframes = []
# Подготовка файлов
manager = FileManagement()
manager.pipeline()

# Подготовка датасета
preprocessor = DatasetPreprocessing()
dataset_dict = preprocessor.pipeline()

# Обучение модели
trainer = ModelTraining(dataset_dict)
trainer.pipeline()

# Инференс модели
producer = Production()
producer.pipeline()