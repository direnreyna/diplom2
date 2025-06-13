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
(x_train, y_train, x_val, y_val, x_test, y_test) = preprocessor.pipeline()

# Обучение модели
trainer = ModelTraining()
trainer.pipeline(x_train, y_train, x_val, y_val, x_test, y_test)

# Инференс модели
producer = Production()
producer.pipeline()