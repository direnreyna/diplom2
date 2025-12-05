# src/dataset_preprocessing

import os
import pandas as pd
from .config import config 
from .dataset_loading import DatasetLoading
from .dataset_analyzer import DatasetAnalyze
from .dataset_filtering import DatasetFiltering
from .dataset_preparing import DatasetPreparing
from typing import TYPE_CHECKING

# Для обещания типизации, чтобы не создавать циклический вызов классов:
# DatasetPreprocessing <-> DatasetFiltering
# DatasetPreprocessing <-> DatasetAnalyze
if TYPE_CHECKING:
    from .dataset_filtering import DatasetFiltering
    from .dataset_analyzer import DatasetAnalyze

class DatasetPreprocessing():
    """Класс, управлящий подготовкой данных"""
    def __init__(self):
        self.temp_dir = config['paths']['temp_dir']
        self.patient_ids = []
        self.stage = ''

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

        self.df_total_signals = pd.DataFrame()                  # Сигналы по весм каналам
        self.df_total_annotations = pd.DataFrame()              # Аннотации по весм каналам

        # === Список сохраненных на диск выборок для моделей ===
        self.dataset_dict = {}                                  # Ключи: 'top' / 'cross' / 'uni_1' / 'uni_2'

        # === Прочие параметры ===
        self.target_channel_name_1 = ''
        self.target_channel_name_2 = ''
        self.channels_per_patient = {}                          # Каналы у пациентов

    def load_dataset(self):
        """Загрузка датасета"""
        loader = DatasetLoading()
        self.df_all_signals, self.df_all_annotations, self.patient_ids = loader.pipeline()
        return self
    
    def filter_data(self) -> 'DatasetPreprocessing':
        """Запускает фильтрацию данных по каналам."""
        filterer: 'DatasetFiltering' = DatasetFiltering(self)
        filterer.run()
        return self
    
    def analyze_dataset(self) -> 'DatasetPreprocessing':
        """Запускает исследовательский анализ данных (EDA)."""
        analyzer: 'DatasetAnalyze' = DatasetAnalyze(self)
        analyzer.run()
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
        self.df_united_annotation_2,
        self.df_total_signals,
        self.df_total_annotations)
        preparer.pipeline()
        return self
        
    def pipeline(self):
        """Полный пайплайн от загрузки до подготовки (X, y)"""
        loader = DatasetLoading()
        if not loader.check_datasets_exists():
           self.load_dataset().filter_data().analyze_dataset().prepare_data()
        
        ### # Временно для теста аналитики
        ### self.load_dataset().filter_data().analyze_dataset()

    def ensure_patient_summary_exists(self):
        """
        Проверяет наличие JSON-файла со сводкой по пациентам.
        Если файла нет, запускает минимальный пайплайн для его создания.
        """
        summary_path = os.path.join(config['paths']['data_dir'], "patient_detailed_summary.json")
        if os.path.exists(summary_path):
            print("Файл сводки по пациентам найден. Создание пропускается.")
            return

        print("Файл сводки по пациентам не найден. Запускаю генерацию...")
        # 1. Загружаем сырые данные
        self.load_dataset()
        # 2. Добавляем информацию о ритме (нужна для анализа)
        filterer = DatasetFiltering(self)
        filterer._add_rhythm_annotations()
        # 3. Создаем анализатор и вызываем ТОЛЬКО нужный метод
        analyzer = DatasetAnalyze(self)
        analyzer._generate_and_save_detailed_summary()