# main.py

import os
import sys
import mlflow

# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.file_management import FileManagement
from src.dataset_preprocessing import DatasetPreprocessing
from src.model_trainer import ModelTraining
from src.model_producer import Production

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Скрывает INFO/WARNING от TensorFlow

import tensorflow as tf
print("GPU доступен:", tf.config.list_physical_devices('GPU'))

# Подготовка файлов
manager = FileManagement()
manager.pipeline()

# Подготовка датасета
preprocessor = DatasetPreprocessing()
preprocessor.pipeline()

# Устанавливаем имя эксперимента
mlflow.set_experiment("ECG R-peak Classification")

# Запускаем сессию отслеживания
with mlflow.start_run():
    # Логируем, с какой стадией и префиксом мы работаем
    mlflow.log_param("stage", "stage1")
    mlflow.log_param("prefix", "top")

    # Обучение модели top
    trainer = ModelTraining('stage1', 'top')
    trainer.pipeline(mode='full')

    # Обучение модели top
    # trainer = ModelTraining('stage3', 'top')
    # trainer.pipeline(mode='full')

    # Оценка модели top
    # trainer = ModelTraining('stage2', 'top')
    # trainer.pipeline(mode='eval')

    # Оценка модели cross
    # trainer = ModelTraining('stage1', 'cross')
    # trainer.pipeline(mode='eval')

    # Обучение модели uni_1
    # trainer = ModelTraining('uni_1')
    # trainer.pipeline(mode='full')

    # Обучение модели uni_2
    # trainer = ModelTraining('uni_2')
    # trainer.pipeline(mode='full')

# Инференс модели
producer = Production()
producer.pipeline()