# main.py

import os
import sys
import mlflow

# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
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
current_stage = 'stage2'                    ## 'stage1', 'stage2'
current_prefix = 'top'                      ## 'top', 'cross', 'uni_1', 'uni_2'
current_mode = 'full'                       ## 'full', 'eval'

# Запускаем сессию отслеживания
with mlflow.start_run():

    load_from_mlflow = False                                ## Включает/выключает загрузку из MLflow
    mlflow_run_id = "246962fa145e4eefa82de1b8848454e6"      ## ID запуска MLflow для загрузки модели

    # Логируем, с какой стадией и префиксом мы работаем
    mlflow.log_param("stage", current_stage)
    mlflow.log_param("prefix", current_prefix)
    mlflow.log_param("attention_type", config['params']['attention_type'])

    # Включаем автологирование для Keras. Оно само будет логировать метрики на каждой эпохе.
    mlflow.keras.autolog(
        log_model_signatures=True,
        log_input_examples=False,
        log_models=True,
        disable=False
    )

    if current_stage in config['stages']['multi']:
        monitor = 'val_f1_score'
    else:
        monitor = 'val_accuracy'

    mlflow.log_param("early_stop_monitor", monitor)
    mlflow.log_param("early_stop_patience", config['params']['patience_early_stop'])
    
    # Логируем параметры ReduceLROnPlateau, так как autolog их не логирует
    mlflow.log_param("reduce_lr_monitor", monitor)
    mlflow.log_param("reduce_lr_factor", config['params']['factor_reduce_lr'])
    mlflow.log_param("reduce_lr_patience", config['params']['patience_reduce_lr'])
    mlflow.log_param("reduce_lr_min_lr", config['params']['min_learning_rate'])

    # Обучение модели current_prefix
    #trainer = ModelTraining(current_stage, current_prefix)
    trainer = ModelTraining(current_stage, current_prefix, load_from_mlflow, mlflow_run_id) 

    trainer.pipeline(mode=current_mode)

# Инференс модели
producer = Production()
producer.pipeline()