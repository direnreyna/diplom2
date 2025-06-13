# main.py

import os
import sys
# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

#import src.config
from config import config

from file_management import FileManagement
from dataset_loading import DatasetLoading
from dataset_analyzer import DatasetAnalyze
from dataset_preparing import DatasetPreparing
from model_trainer import ModelTraining
from model_producer import Production

# Подготовка файлов
manager = FileManagement()
manager.pipeline()

# Загрузка датасета
loader = DatasetLoading()
df_x, df_y = loader.pipeline()

# Анализ датасета
analyzer = DatasetAnalyze(df_x, df_y)
(df_top_signals, df_top_annotations, 
df_cross_signals, df_cross_annotations, 
df_united_signals_1, df_united_annotation_1, 
df_united_signals_2, df_united_annotation_2) = analyzer.pipeline()

# Подготовка датасета
preparer = DatasetPreparing(
    df_top_signals, df_top_annotations, 
	df_cross_signals, df_cross_annotations, 
	df_united_signals_1, df_united_annotation_1, 
	df_united_signals_2, df_united_annotation_2)
(x_train, y_train, x_val, y_val, x_test, y_test) = preparer.pipeline()

# Обучение модели
trainer = ModelTraining()
trainer.pipeline(x_train, y_train, x_val, y_val, x_test, y_test)

# Инференс модели
producer = Production()
producer.pipeline()