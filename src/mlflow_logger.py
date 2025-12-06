# src/mlflow_logger.py

import mlflow
from .config import config

class MLFlowLogging:
    """
    Автономный класс для инкапсуляции всей логики логирования в MLflow.
    Использует статический метод, не требует создания экземпляра.
    """

    @staticmethod
    def setup_logging() -> None:
        """Основной статический метод для запуска всех этапов логирования."""
        # Логируем, с какой стадией и префиксом мы работаем
        mlflow.log_param("stage", config['execution']['stage'])
        mlflow.log_param("prefix", config['execution']['prefix'])
        mlflow.log_param("attention_type", config['params']['attention_type'])

        # --- Гиперпараметры модели и данные ---
        mlflow.log_params(config['params'])
        mlflow.log_params(config['data'])

        # --- Параметры коллбэков, которые не логируются автологом ---
        if config['execution']['stage'] in config['stages']['multi']:
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

        # Включаем автологирование для Keras. Оно само будет логировать метрики на каждой эпохе.
        mlflow.keras.autolog(
            log_model_signatures=True,
            log_input_examples=False,
            log_models=True,
            disable=False
        )