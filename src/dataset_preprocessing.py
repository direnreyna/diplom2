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

        # === Список сохраненных на диск выборок для моделей ===
        self.dataset_dict = {}                                  # Ключи: 'top' / 'cross' / 'uni_1' / 'uni_2'

        # === Прочие параметры ===
        self.target_channel_name_1 = ''
        self.target_channel_name_2 = ''

    def load_dataset(self):
        """Загрузка датасета"""
        loader = DatasetLoading()
        self.df_all_signals, self.df_all_annotations, self.patient_ids = loader.pipeline()
        return self
    
    def analyze_channels(self):
        """Анализ датасета"""
        analyzer = DatasetAnalyze(self.df_all_signals, self.df_all_annotations, self.patient_ids)
        (self.target_channel_name_1,
        self.target_channel_name_2,
        self.df_top_signals,
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
        preparer = DatasetPreparing(
        self.target_channel_name_1,
        self.target_channel_name_2,
        self.df_top_signals,
        self.df_top_annotations, 
        self.df_cross_signals, 
        self.df_cross_annotations, 
        self.df_united_signals_1, 
        self.df_united_annotation_1, 
        self.df_united_signals_2, 
        self.df_united_annotation_2)
        self.dataset_dict = preparer.pipeline()
    
    def pipeline(self):
        """Полный пайплайн от загрузки до подготовки (X, y)"""
        loader = DatasetLoading()
        if not loader.check_datasets_exists():
            ## Временно для теста аналитики
            self.load_dataset().analyze_channels().prepare_data()
            ## self.load_dataset().analyze_channels()