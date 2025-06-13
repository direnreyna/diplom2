# dataset_preprocessing

import pandas as pd
from config import config

from dataset_loading import DatasetLoading
from dataset_analyzer import DatasetAnalyze
from dataset_preparing import DatasetPreparing

class DatasetPreprocessing():
    """Класс, управлящий подготовкой данных"""
    def __init__(self):
        self.temp_dir = config['paths']['temp_dir']
        self.patient_ids = []

        # === Данные до фильтрации ===
        self.df_all_signals = pd.DataFrame()
        self.df_all_annotations = pd.DataFrame()

        # === Фильтрация по каналам ===
        self.df_top_signals = pd.DataFrame()                    # Отфильтрованные сигналы для формирования ДС по 1 топ-каналу
        self.df_top_annotations = pd.DataFrame()                # Аннотации для формирования ДС по 1 топ-каналу

        self.df_cross_signals = pd.DataFrame()                  # Отфильтрованные сигналы для формирования кросс-теста по не топ-каналу
        self.df_cross_annotations = pd.DataFrame()              # Аннотации для формирования кросс-теста по не топ-каналу

        self.df_united_signals_1 = pd.DataFrame()               # Отфильтрованные сигналы для формирования 1го ДС по 2 топ-каналам
        self.df_united_annotation_1 = pd.DataFrame()            # Аннотации для формирования 1го ДС по 2 топ-каналам

        self.df_united_signals_2 = pd.DataFrame()               # Отфильтрованные сигналы для формирования 2го ДС по 2 топ-каналам
        self.df_united_annotation_2 = pd.DataFrame()            # Аннотации для формирования 2го ДС по 2 топ-каналам

        # === Окончательные выборки для модели ===
        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None

        self.x_test = None
        self.y_test = None

        # === Прочие параметры ===
        self.target_channel_name_1 = ''
        self.target_channel_name_2 = ''

    def load_dataset(self):
        """Загрузка датасета"""
        loader = DatasetLoading()
        self.df_all_signals, self.df_all_annotations = loader.pipeline()
        return self
    
    def analyze_channels(self):
        """Анализ датасета"""
        analyzer = DatasetAnalyze(self.df_all_signals, self.df_all_annotations)
        (self.df_top_signals,
        self.df_top_annotations, 
        self.df_cross_signals, 
        self.df_cross_annotations, 
        self.df_united_signals_1, 
        self.df_united_annotation_1, 
        self.df_united_signals_2, 
        self.df_united_annotation_2) = analyzer.pipeline()
        return self
    
    def prepare_data(self):
        """Формирование окон, меток, разделение на train/val/test"""
        preparer = DatasetPreparing(self.df_top_signals,
        self.df_top_annotations, 
        self.df_cross_signals, 
        self.df_cross_annotations, 
        self.df_united_signals_1, 
        self.df_united_annotation_1, 
        self.df_united_signals_2, 
        self.df_united_annotation_2)
        (x_train, y_train, x_val, y_val, x_test, y_test) = preparer.pipeline()

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        
        return self
    
    def pipeline(self):
        """Полный пайплайн от загрузки до подготовки (X, y)"""
        self.load_dataset().analyze_channels().prepare_data()
        return (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test)
