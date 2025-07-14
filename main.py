# main.py

import os
import sys
import numpy as np
import pandas as pd
import mlflow
#from typing import pd.DataFrame

# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.mlflow_logger import MLFlowLogging
from src.file_management import FileManagement
from src.dataset_preprocessing import DatasetPreprocessing
from src.model_trainer import ModelTraining
from src.window_inferencer import WindowInference 
#from src.model_producer import Production
from src.dataset_preparing import DatasetPreparing

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


# --- ДИАГНОСТИЧЕСКИЙ БЛОК (ВЕРСИЯ 2) --- ## ИЗМЕНЕНА ВСЯ СЕКЦИЯ
print("\n" + "="*50)
print("ЗАПУСК ГЛУБОКОЙ ДИАГНОСТИКИ СИНХРОНИЗАЦИИ ДАННЫХ STAGE2")
print("="*50)

# 1. Загружаем "проблемный" датасет
problem_dataset_path = os.path.join(config['paths']['data_dir'], 'top_stage2_dataset.npz')
if os.path.exists(problem_dataset_path):
    data = np.load(problem_dataset_path, allow_pickle=True)
    y_test_problem = data['y_test']
    metadata_test_problem = data['metadata_test']
    
    # 2. Загружаем ПЕРВОИСТОЧНИК аннотаций заново, чтобы гарантировать его чистоту
    # Используем готовый класс DatasetLoading
    from src.dataset_loading import DatasetLoading
    from src.dataset_filtering import DatasetFiltering
    
    print("Перезагрузка исходных данных для сверки...")
    loader_for_debug = DatasetLoading()
    _, df_annotations_source, _ = loader_for_debug.pipeline()
    
    # Добавляем к нему колонку Current_Rhythm, так как она нужна для _is_label_valid_for_stage
    # Создаем временный контейнер для этого
    class TmpContainer:
        def __init__(self, df):
            self.df_all_annotations = df
            self.patient_ids = df['Patient_id'].unique()
    
    container = TmpContainer(df_annotations_source)
    filterer_for_debug = DatasetFiltering(container)
    filterer_for_debug._add_rhythm_annotations()
    df_annotations_source = filterer_for_debug.container.df_all_annotations
    
    # 3. Создаем экземпляр preparer, чтобы получить доступ к логике меток
    preparer_for_test = DatasetPreparing(None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    preparer_for_test.stage = 'stage2'

    mismatch_count = 0
    num_samples_to_check = 100

    print(f"Проверка первых {num_samples_to_check} примеров из тестовой выборки stage2...")
    for i in range(num_samples_to_check):
        meta_tuple = tuple(metadata_test_problem[i])
        label_from_y_test = np.argmax(y_test_problem[i])

        p_id, s_id = str(meta_tuple[0]), int(meta_tuple[1])
        original_row_series = df_annotations_source[
            (df_annotations_source['Patient_id'] == p_id) & 
            (df_annotations_source['Sample'] == s_id)
        ]
        
        if original_row_series.empty:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Пик {meta_tuple} из датасета не найден в первоисточнике!")
            continue

        original_row = original_row_series.iloc[0]
        ground_truth_label = preparer_for_test.is_label_valid_for_stage(original_row)
        
        if label_from_y_test != ground_truth_label:
            mismatch_count += 1
            print("-" * 20)
            print(f"!!! НАЙДЕНО НЕСООТВЕТСТВИЕ №{mismatch_count} для пика {meta_tuple} !!!")
            print(f"    Метка в y_test:      {label_from_y_test}")
            print(f"    Метка из источника:  {ground_truth_label} (на основе Type='{original_row['Type']}', Rhythm='{original_row['Current_Rhythm']}')")
    
    print("-" * 50)
    if mismatch_count == 0:
        print("ПРОВЕРКА ПРОЙДЕНА: Несоответствий не найдено. Данные X_test и y_test синхронизированы.")
    else:
        print(f"ПРОВЕРКА ПРОВАЛЕНА: Найдено {mismatch_count} несоответствий в {num_samples_to_check} примерах.")
        print("Это доказывает, что проблема в рассинхронизации данных при их подготовке.")

else:
    print(f"Диагностика невозможна: файл {problem_dataset_path} не найден.")