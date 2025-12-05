# main.py

"""
Главный скрипт для запуска полного пайплайна проекта.

Выполняет последовательно следующие шаги:
1. Подготовка файлов (копирование из input в temp).
2. Подготовка датасета (загрузка, фильтрация, EDA, создание .npz файлов).
   Этот шаг пропускается, если датасеты уже созданы.
3. Запуск сессии MLflow для отслеживания эксперимента.
4. Обучение или оценка модели в зависимости от параметра 'mode' в config.yaml.
5. Запуск демонстрационного инференса для одного случайного R-пика.

Предназначен для запуска из командной строки: `python main.py`
"""

import os
import sys
import mlflow

# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
#sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.mlflow_logger import MLFlowLogging
from src.file_management import FileManagement
from src.dataset_preprocessing import DatasetPreprocessing
from src.model_trainer import ModelTraining
from src.window_inferencer import WindowInference 

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

# Устанавливаем значения эксперимента
mlflow.set_experiment("ECG R-peak Classification")

# Запускаем сессию отслеживания
with mlflow.start_run():

    # Настраиваем логирование
    MLFlowLogging.setup_logging()

    # Обучение модели current_prefix
    trainer = ModelTraining(
        config['execution']['stage'],
        config['execution']['prefix'],
        config['execution']['load_from_mlflow'],
        config['execution']['mlflow_run_id'].get(config['execution']['stage'])
        ) 
    trainer.pipeline(mode=config['execution']['mode'])

# Инференс модели
inferencer = WindowInference(prefix=config['execution']['prefix'])
inferencer.predict_random()