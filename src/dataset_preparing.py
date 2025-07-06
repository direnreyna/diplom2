# src/dataset_preparing

import os
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from config import config
from typing import Tuple, Dict, Union, List, Mapping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from tabulate import tabulate

class DatasetPreparing:
    def __init__(self,
        target_channel_name_1: str,
        target_channel_name_2: str,
        df_top_signals: pd.DataFrame,
        df_top_annotations: pd.DataFrame, 
	    df_cross_signals: pd.DataFrame,
        df_cross_annotations: pd.DataFrame, 
	    df_united_signals_1: pd.DataFrame,
        df_united_annotation_1: pd.DataFrame, 
	    df_united_signals_2: pd.DataFrame,
        df_united_annotation_2: pd.DataFrame,
        df_total_signals: pd.DataFrame,
        df_total_annotations: pd.DataFrame
        ) -> None:

        self.all_stages = config['stages']['all']
        self.multi_stages = config['stages']['multi']

        self.num_classes: int                                   # Количество классов для мультикллассовой классификации
        self.target_channel_name_1 = target_channel_name_1      # Имя основного канала для обучения (например, 'MLII')     
        self.target_channel_name_2 = target_channel_name_2      # Имя второго канала для сравнения (например, 'V1')
            
        self.df_top_signals = df_top_signals                    # Отфильтрованные сигналы для формирования ДС по 1 топ-каналу
        self.df_top_annotations = df_top_annotations            # Аннотации для формирования ДС по 1 топ-каналу
        self.df_cross_signals = df_cross_signals                # Отфильтрованные сигналы для формирования кросс-теста по не топ-каналу
        self.df_cross_annotations = df_cross_annotations        # Аннотации для формирования кросс-теста по не топ-каналу
        self.df_united_signals_1 = df_united_signals_1          # Отфильтрованные сигналы для формирования 1го ДС по 2 топ-каналам
        self.df_united_annotation_1 = df_united_annotation_1    # Аннотации для формирования 1го ДС по 2 топ-каналам
        self.df_united_signals_2 = df_united_signals_2          # Отфильтрованные сигналы для формирования 2го ДС по 2 топ-каналам
        self.df_united_annotation_2 = df_united_annotation_2    # Аннотации для формирования 2го ДС по 2 топ-каналам
        self.df_total_signals = df_total_signals                # Сигналы по весм каналам
        self.df_total_annotations = df_total_annotations        # Аннотации по весм каналам

    def pipeline(self) -> None:
        """
        Перебираем все стадии проекта и проверяем нужно ли создавать и сохраныть датасет для обучения модели
        """
        for stage in self.all_stages:
            self._check_create_and_save(self.df_top_signals, self.df_top_annotations, self.target_channel_name_1, stage, 'top')
            self._check_create_and_save(self.df_cross_signals, self.df_cross_annotations, self.target_channel_name_2, stage, 'cross')
            self._check_create_and_save(self.df_united_signals_1, self.df_united_annotation_1, self.target_channel_name_1, stage, 'uni1')
            self._check_create_and_save(self.df_united_signals_2, self.df_united_annotation_2, self.target_channel_name_2, stage, 'uni2')
            #self._check_create_and_save(self.df_total_signals, self.df_total_annotations, 'Signal', stage, 'total')

    def _check_create_and_save(self, df_signals:pd.DataFrame, df_annotations:pd.DataFrame, channel:str, stage:str, prefix:str):
        """
        Проверяет наличие сохраненного набора датасетов для обучения и валидации модели
        Если не находит: создает и сохраняет (дифференцированно по каждому датасету)
        """
        if not self._is_exsist(stage, prefix):
            X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top = self._create_dataset(df_signals, df_annotations, channel, stage, prefix)
            self._save_dataset(X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top, prefix, stage)
            del X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top

    def _is_exsist(self, stage:str, prefix:str) -> bool:
        """
        Проверяет наличие на диске сохраненного датасета для выбранной стадии
        """
        dataset_name = config['data']['dataset_name']
        dataset_path = config['paths']['data_dir']
        file_dataset = os.path.join(dataset_path, f"{prefix}_{stage}_{dataset_name}")

        print(f"Проверяю наличие сохраненного датасета: [{file_dataset}]")
        if os.path.exists(file_dataset):
            print("Датасет найден. Пропускаю создание.")
            return True
        else:
            print("Датасет не найден. Создаю...")
            return False

    def _create_dataset(self, df_sign: pd.DataFrame, df_anno: pd.DataFrame, channel: Union[str, Tuple[str]], stage: str, prefix:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Формирует обучающий/валидационный/тестовый датасет из исходных сигналов и аннотаций.
        Включает создание окон, добавление производных, разделение на выборки,
        аугментацию, нормализацию и перевод меток в OHE-формат при необходимости.

        :param df_sign: pd.DataFrame с сигналами, содержит колонки ['Patient_id', 'Sample', 'Signal', ...]
        :param df_anno: pd.DataFrame с аннотациями, содержит колонки ['Patient_id', 'Sample', 'Type', 'Current_Rhythm']
        :param channel: название канала (например, 'MLII') или кортеж названий каналов.
                        Может быть строкой или кортежем для мультиканального входа.
        :param stage: текущая стадия обучения (используется для фильтрации меток и применения правил)
        
        :return: шесть массивов:
                - X_train_norm: нормализованные обучающие данные
                - y_train_with_aug: метки обучающей выборки с аугментациями
                - X_val_norm: нормализованные валидационные данные
                - y_val: метки валидационной выборки
                - X_test_norm: нормализованные тестовые данные
                - y_test: метки тестовой выборки        
        """
        self.stage = stage
        self.prefix = prefix    
        
        print("\n[DEBUG] ДО _create_windows_and_labels")
        print("df_signals.head():\n", df_sign.head())
        print("df_annotations.head():\n", df_anno.head())
        print("df_signals.shape:", df_sign.shape)
        print("df_annotations.shape:", df_anno.shape)
        print("channel:", channel, "| type:", type(channel))
        print("stage:", stage, "| type:", type(stage))
        print("prefix:", self.prefix, "| type:", type(self.prefix))

        # Формируем окна
        X_windowed, y = self._create_windows_and_labels(df_sign, df_anno, channel)

        ### ##### Блок дебаггинга
        ### if len(y) == 0:
        ###     raise ValueError("Сформированные метки пусты. Ошибка на этапе создания окон.")        
        ### unique_classes = np.unique(y)
        ### print(f">>> В датасете присутствуют классы: {unique_classes}")
        ### if self.stage in self.multi_stages and len(unique_classes) < 2:
        ###     raise ValueError("На многоклассовой стадии недостаточно уникальных меток для обучения.")
        ### ##### Блок дебаггинга

        print("\n[AFTER] _create_windows_and_labels")
        print("X_windowed[:5]:\n", X_windowed[:5])
        print("X_windowed.shape:", X_windowed.shape)
        print("X_windowed.dtype:", X_windowed.dtype if isinstance(X_windowed, np.ndarray) else 'list')
        print("y[:5]:", y[:5])
        print("y.shape:", np.array(y).shape)
        print("y.unique:", np.unique(y))

        # Добавляем производные
        X_derivated = self._add_derivatives_to_windows(X_windowed)

        print("\n[AFTER] _add_derivatives_to_windows")
        print("X_derivated[:5]:\n", X_derivated[:5])
        print("X_derivated.shape:", X_derivated.shape)
        print("X_derivated.dtype:", X_derivated.dtype)

        # Разделяем на выборки
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_dataset(X_derivated, y)

        print("\n[AFTER] _split_dataset")
        print("X_train[:5]:\n", X_train[:5])
        print("X_train.shape:", X_train.shape)
        print("y_train[:5]:", y_train[:5])
        print("y_train.shape:", y_train.shape)
        print("y_train.unique:", np.unique(y_train))

        # Добавляем аугментацию в обучающую выборку
        X_train_with_aug, y_train_with_aug  = self._x_train_augmentation(X_train, y_train)

        print("\n[AFTER] _x_train_augmentation")
        print("X_train_with_aug[:5]:\n", X_train_with_aug[:5])
        print("X_train_with_aug.shape:", X_train_with_aug.shape)
        print("y_train_with_aug[:5]:", y_train_with_aug[:5])
        print("y_train_with_aug.shape:", y_train_with_aug.shape)
        print("y_train_with_aug.unique:", np.unique(y_train_with_aug))

        # Если мультикласс, то метки переводим в ohe-формат
        y_train_with_aug, y_val, y_test = self._convert_to_ohe(y_train_with_aug, y_val, y_test)

        print("\n[AFTER] _convert_to_ohe")
        print("y_train_with_aug[:5]:\n", y_train_with_aug[:5])
        print("y_train_with_aug.shape:", y_train_with_aug.shape)
        print("y_val[:5]:", y_val[:5])
        print("y_test[:5]:", y_test[:5])

        # Нормализуем окна
        X_train_norm, X_val_norm, X_test_norm = self._normalize_windows(X_train_with_aug, X_val, X_test)

        print("\n[AFTER] _normalize_windows")
        print("X_train_norm[:5]:\n", X_train_norm[:5])
        print("X_train_norm.shape:", X_train_norm.shape)
        print("X_train_norm.dtype:", X_train_norm.dtype)
        print("X_val_norm.shape:", X_val_norm.shape)
        print("X_test_norm.shape:", X_test_norm.shape)

        print("Выборка сформирована")
        return X_train_norm, y_train_with_aug, X_val_norm, y_val, X_test_norm, y_test

    def _is_label_valid_for_stage(self, row:Union[pd.Series, str]) -> Union[int, None]:
        """Формируем метку в зависимости от стадии"""
        
        if isinstance(row, str):
            for_check = row
            for_check2 = row # для строки не подается 2й параметр
        else:
            for_check = row['Type']
            for_check2 = row['Current_Rhythm']

        if self.stage == 'stage1':
            if for_check == 'N' and for_check2 == 'N':
                return 0  # "Good" (53%)
            else:
                return 1  # "Alert" (47%)

        elif self.stage == 'stage(не реализованная)':
            if for_check == 'N' and for_check2 == 'N':
                return None         # отсев на 1й стадии
            elif for_check == 'N' and for_check2 != 'N':
                return 0  # "Attention" (29%)
            else:
                return 1  # "Alarm" (71%)

        elif self.stage == 'stage_all(не реализованная)':
            self.num_classes = 11
            if for_check == 'N':
                return 0  # N: 68.98% == суперкласс Normal
            elif for_check == 'L':
                return 1  # L: 15.25%
            elif for_check == 'R':
                return 2  # R: 13.71%
            elif for_check == 'A':
                return 3  # A: 4.81%
            elif for_check == 'a':
                return 4  # a: 0.28%
            elif for_check == 'J':
                return 5  # J: 0.16%
            elif for_check == 'e':
                return 6  # e: 0.03%
            elif for_check == 'j':
                return 7  # j: 0.43%
            elif for_check == 'V':
                return 8  # V: 13.47%
            elif for_check == 'E':
                return 9  # E: 0.20%
            elif for_check == 'F':
                return 10  # F: 1.52%
            elif for_check == '+':
                return 11  # +: 2.44%
            elif for_check in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 12  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage2a':
            self.num_classes = 11
            if for_check == 'N' and for_check2 == 'N':
                return None         # отсев на 1й стадии
            elif for_check == 'N' and for_check2 != 'N':
                return 0  # N: 28.98% == суперкласс Normal

            elif for_check == 'L':
                return 1  # L: 15.25%
            elif for_check == 'R':
                return 2  # R: 13.71%
            elif for_check == 'A':
                return 3  # A: 4.81%
            elif for_check == 'a':
                return 4  # a: 0.28%
            elif for_check == 'J':
                return 5  # J: 0.16%
            elif for_check == 'e':
                return 6  # e: 0.03%
            elif for_check == 'j':
                return 7  # j: 0.43% == суперкласс SVEB

            elif for_check in ['V', 'E']:
                return 8  # V: 13.47%, E: 0.20% == суперкласс VEB: 13.67%

            elif for_check in ['F', '+']:
                return 9  # F: 1.52%, +: 2.44% == суперкласс Fusion: 3.96%

            elif for_check in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 10  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage2':
            self.num_classes = 7
            if for_check == 'N' and for_check2 == 'N':
                return None         # отсев на 1й стадии
            elif for_check == 'N' and for_check2 != 'N':
                return 0  # N: 28.98% == суперкласс Normal

            elif for_check == 'L':
                return 1  # L: 15.25% == LBBB
            elif for_check == 'R':
                return 2  # R: 13.71% == RBBB
            elif for_check in ['A', 'a', 'J', 'e', 'j']:
                return 3  # A: 4.81%, a: 0.28%, J: 0.16%, e: 0.03%, j 0.43% == суперкласс subSVEB

            elif for_check in ['V', 'E']:
                return 4  # V: 13.47%, E: 0.20% == суперкласс VEB: 13.67%

            elif for_check in ['F', '+']:
                return 5  # F: 1.52%, +: 2.44% == суперкласс Fusion: 3.96%

            elif for_check in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 6  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage3':
            self.num_classes = 5

            if for_check in ['A']:
                return 0  # A: 4.81%
            elif for_check in ['a']:
                return 1  # a: 0.28%
            elif for_check in ['J']:
                return 2  # J: 0.16%
            elif for_check in ['e']:
                return 3  # e: 0.03%
            elif for_check in ['j']:
                return 4  # j 0.43%
            else:
                return None

        elif self.stage == 'stage01':
            self.num_classes = 13
            if for_check == 'N':
                return 0  # N: 68.98% == суперкласс Normal
            elif for_check == 'L':
                return 1  # L: 15.25%
            elif for_check == 'R':
                return 2  # R: 13.71%
            elif for_check == 'A':
                return 3  # A: 4.81%
            elif for_check == 'a':
                return 4  # a: 0.28%
            elif for_check == 'J':
                return 5  # J: 0.16%
            elif for_check == 'e':
                return 6  # e: 0.03%
            elif for_check == 'j':
                return 7  # j: 0.43%
            elif for_check == 'V':
                return 8  # V: 13.47%
            elif for_check == 'E':
                return 9  # E: 0.20%
            elif for_check == 'F':
                return 10  # F: 1.52%
            elif for_check == '+':
                return 11  # +: 2.44%
            elif for_check in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']', '|']:
                return 12  # шумы: 18.45% == суперкласс Q
            else:
                return None
            
    def _create_windows_and_labels(self,
        df_signals: pd.DataFrame,
        df_annotations: pd.DataFrame,
        channels: Union[str, Tuple[str]]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формирует окна вокруг R-пиков
        
        :param df_signals: pd.DataFrame с Sample и каналом target_channel
        :param df_annotations: pd.DataFrame с Sample и метками Type + Current_Rhythm
        :param target_channel: str — имя канала, например 'MLII'. Или 'Signal' для формата 'Total'
        :return: numpy array (X), numpy array (y)
        """
        if isinstance(channels, str):
            channels = [channels]
    
        window_size = config['data']['window_size']  # например, 360
        half_window = window_size // 2
        shift_size = config['data']['aug_shift_size']  # например, 20 (сколько будет создано доп окон скольжением)
        half_shift_size = shift_size // 2
        x_win = []
        y = []
        x_aug_win = defaultdict(lambda: defaultdict(list))
        y_aug = defaultdict(lambda: defaultdict(list))
        x_aug_total = []
        y_aug_total = []
        df_p_ch_signal = None
        
        if 'Channel' in df_signals:
            df_p_ch_signal = df_signals.groupby(['Patient_id', 'Channel'])

        for target_channel in channels:
            ### print(f"target_channel: {target_channel} в {channels}")
    
            for pid in tqdm(df_annotations['Patient_id'].unique(), desc="Формируем окна"):
                # Выбираем данные пациента
                
                df_p_signal = df_signals[df_signals['Patient_id'] == pid]
                df_p_annotation = df_annotations[df_annotations['Patient_id'] == pid]

                for _, row in df_p_annotation.iterrows():
                    sample = row['Sample']
                    start = sample - half_window
                    end = sample + half_window

                    start_canvas = start - half_shift_size
                    end_canvas = end + half_shift_size

                    if 'Channel' in row:
                        channel_type = row['Channel'] # 
                        # Выбираем часть ДФ, где сгруппированы данные пациента "pid" по каналу "channel_type"
                        df_temp = df_p_ch_signal.get_group((pid, channel_type))

                        # Извлекаем участок сигнала
                        window = df_temp[(df_temp['Sample'] >= start) & (df_temp['Sample'] < end)]
                        # Извлекаем участок для формирования ряда аугментированных шифтингом сигналов
                        aug_canvas = df_temp[(df_temp['Sample'] >= start_canvas) & (df_temp['Sample'] < end_canvas)]
                    else:
                        # Извлекаем участок сигнала
                        window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end)]
                        aug_canvas = df_p_signal[(df_p_signal['Sample'] >= start_canvas) & (df_p_signal['Sample'] < end_canvas)]

                    # Избавляемся от неполных окон по краям набора данных
                    if len(window) != window_size:
                        continue

                    ###############################################################
                    # ФОРМИРОВАНИЕ НАБОРА ( LABELS ) 
                    # Проверяем валидность метки для стадии
                    ###############################################################
                    label = self._is_label_valid_for_stage(row)

                    if label is None:
                        continue

                    signal_values = window[target_channel].values

                    ###############################################################
                    # АУГМЕНТАЦИЯ
                    # Полуение списка аугментированных копий для signal_values
                    ###############################################################
                    label_type = row['Type']
                    channel_type = 'MLII'
                    if 'Channel' in row:
                        channel_type = row['Channel']

                    if label_type != 'N':

                        label = self._is_label_valid_for_stage(label_type) # обозначение Номера класса

                        self.size_augmentations = config['data']['size_augmentations']
                        
                        if len(x_aug_win[channel_type][label]) < self.size_augmentations:
                            print(f"Для метки [{label}] по каналу [{channel_type}] аугментировано {len(x_aug_win[channel_type][label])} копий. Начал аугментацию для {sample}...")
                            aug_signal_values = self._shift_noise_augmentation(
                                np.array(aug_canvas[target_channel].values),
                                np.array(signal_values)
                                )
                            x_aug_win[channel_type][label].extend(aug_signal_values)
                            y_aug[channel_type][label].extend([label] * len(aug_signal_values))      ## добавляем список меток по числу аугментированных копий
                            print(f"Закончил аугментацию для {sample}. Для метки [{label_type}] аугментировано {len(x_aug_win[channel_type][label])} копий.")
                    ###############################################################
                    
                    y.append(label)
                    x_win.append(signal_values)

        # Обрабатываем собранные аугментации по каналам и меткам с выводом статистики
        x_aug_total, y_aug_total = self._process_augmentations(x_aug_win, y_aug, x_aug_total, y_aug_total)

        # Сохраняем на диск аугментации по каждой метке для текущей стадии
        self._save_augmentations_per_labels(x_aug_total, y_aug_total)

        if len(x_win) == 0:
            raise ValueError("Не удалось создать ни одного окна. Возможно, неверный размер окна или фильтрация слишком жёсткая.")
        if len(np.unique(y)) < 2:
            raise ValueError("Недостаточно уникальных меток для обучения модели.")

        return np.array(x_win), np.array(y)

    def _shift_noise_augmentation(self, 
            canvas: np.ndarray,         ## полотно для нарезки на аугментированные окна шифтингом
            orig_win: np.ndarray        ## оригинальное окно для аугментирования
            ) -> List[np.ndarray]:
        """
        Метод создает до 10 до 20 аугментированных копий для оригинала.
        Копии создаются методом сдвига окна на 10 разных позиций влево и на 10 разных позиций вправо.
        Если оригианальное окно находится на границе данных, то как минимум с одной из сторон будет создано 10 аугментированных копий.
        """
        win_size = len(orig_win)

        augmented_windows = []
        # canvas гарантированно длиннее orig_win минимум на half_shift_size с одной из сторон
        for i in range(len(canvas) - win_size): ## == от shift_size // 2 до shift_size (от 10 до 20)
            shifted_window = canvas[i:win_size + i]
            
            # Избегаем дублирования оригинала
            if np.array_equal(shifted_window, orig_win):
                continue
            # Избавляемся от неполных окон по краям набора данных
            if len(shifted_window) != win_size:
                continue
            
            # augmented_windows.append(shifted_window)       ## Добавлять в итоговый список только сдвинутые окна, без шумов
            # создаем аугментированные копии сдвинутых окон с шумами
            noised_windows = self._noise_augmentation(shifted_window)
            augmented_windows.extend(noised_windows)

        return augmented_windows

    def _noise_augmentation(self, shifted_window: np.ndarray) -> List[np.ndarray]:
        """
        Создает аугментированные копии окна добавлением шума в распределенные 18 позиций
        Каждая позиция сучайно определяется в своём диапазоне [0:20), [20:40) и т.д.
        Во все указанные позиции будет добавлен шум:
         - либо мелкий (+0.02 .. +0.08),
         - либо крупный (+0.12 .. +0.18).
        Наличие крупного шума определяется контролируемым перебором 12 из 18 позиций во вложенном цикле.
        Шум крупный подобран так, чтобы не исчезнуть при нормализации.
        Крупный шусм нужен для создания уникальности аугментированных копий
        Мелкий нужен для вариативности, но его исчезновение не создаст дублей.

        """
        window_size = len(shifted_window)
        noised_windows = []

        ## random_shift_range - величина отклонения позиции от базовых (равномерно распределенных) позиций по окну
        rsr = 10
        ## список базовых позиций (18 штук), возле которых могут располагаться точки появления шума
        noise_positions = list(range(rsr, window_size - rsr, rsr * 2))
        ## список случайно сдвинутых (относительно базовых) позиций, где будут располагаться точки появления шума
        shifted_noise_positions = [shifted_pos + int(np.random.randint(-rsr, rsr)) for shifted_pos in noise_positions]

        # Создание 3 непересекающихся групп позиций: penta_zone_pos, penta_zone_pos-1, penta_zone_pos-2
        penta_zone_pos = np.array([2, 5, 8, 11, 14])
        # Используя 3 вложенных цикла генерируем 125 (5*5*5) уникальных копий (new_window) на каждое shifted_window
        # Для того, чтобы вариативность шумов была разнообразнее, каждый цикл отключает 2 подконтрольных ему позиции
        # (позиции в разных циклах не пересекаются). Всего будет добавлено 12 крупных шумов и 6 мелких.
        for pos1 in penta_zone_pos:                        ## 5 копий
            for pos2 in penta_zone_pos-1:                  ## 25 = 5*5 копий
                for pos3 in penta_zone_pos-2:              ## 125 = 5*5*5 копий
                    
                    # Список позиций отключаемых от контроллируемой аугнментации шумами
                    selected = [pos1, pos2, pos3, pos1 + 3, pos2 + 3, pos3 + 3]  # позиции "+3" нужны для того, чтобы вариативность шумов была разнообразнее
                    new_window = shifted_window.copy()
                    
                    # Перебираем номера позиций для размещения шума
                    for check_pos in range(len(shifted_noise_positions)):
                        
                        # мелкий шум в точке shifted_noise_positions[check_pos]: +0.02 .. +0.08
                        if check_pos in selected:
                            new_window[shifted_noise_positions[check_pos]] = shifted_window[shifted_noise_positions[check_pos]] + np.random.uniform(0.02, 0.08)

                        # контролируемая аугментация с мелким шумом в точке shifted_noise_positions[check_pos]: +0.12 .. +0.18
                        else:
                            new_window[shifted_noise_positions[check_pos]] = shifted_window[shifted_noise_positions[check_pos]] + 0.1 + np.random.uniform(0.02, 0.08)
                    noised_windows.append(new_window)
        return noised_windows

    def _add_derivatives_to_windows(self, X:np.ndarray) -> np.ndarray:
        """
        Добавляет к каждому окну первую и вторую производную в каждой точке, создавая 2 доп. слоя
        
        :param X: numpy array of shape (n_samples, window_size)
        :return: numpy array of shape (n_samples, window_size, n_features)
        """

        if config['data']['add_derivatives']:
            X_with_derivatives = []
            for signal in X:
                d1 = np.gradient(signal)
                d2 = np.gradient(d1)
                combined = np.stack([signal, d1, d2], axis=-1)  # (window_size, 3)
                X_with_derivatives.append(combined)
        else:
            X_with_derivatives = X

        return np.array(X_with_derivatives)

    def _split_dataset(self, X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разделяет данные на train/val/test с сохранением баланса классов
        
        :param X: numpy array
        :param y: numpy array
        :param test_size: float
        :param val_size: float
        :return: X_train, y_train, X_val, y_val, X_test, y_test
        """
        test_size=config['data']['test_size']
        val_size=config['data']['val_size']

        # Сначала выделим тест
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            shuffle=True,
            random_state=42
        )

        # Теперь из train выделим val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size / (1 - test_size),
            stratify=y_train,
            shuffle=True,
            random_state=42
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _x_train_augmentation(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применяет аугментации созданные и сохраненные ранее если в config.yaml сказано, что:
        - нужно аугментировать
        - есть список аугментаций по классам для текущей стадии
        
        :parameters: X_train, y_train
        :returns: X_train_with_aug, y_train_with_aug
        """
        X_train_with_aug = X_train.copy()
        y_train_with_aug = y_train.copy()

        # Проверяем. разрешены ли аугментации
        if config['data']['add_augmentations']:
            dir = config['paths']['data_dir']

            # Узнаем. есть ли список аугментаций для этой стадии
            if self.stage in config.get('augs', {}):
                
                print(f"Оригинальных примеров: {X_train.shape[0]}")
                # Читаем список аугментаций с конфига
                for i, aug_quantity in enumerate(config['augs'][self.stage]):
                    
                    if aug_quantity != 0:
                        filename = os.path.join(dir, f"label_{i}_for_{self.stage}.pkl")
                        x_aug, y_aug = joblib.load(filename)

                        # Добавление в X_train "aug_quantity" аугментаций по метке "i"
                        X_train_with_aug = np.concatenate([X_train_with_aug, x_aug[:aug_quantity]], axis=0)
                        y_train_with_aug = np.concatenate([y_train_with_aug, y_aug[:aug_quantity]], axis=0)
                        print(f"По метке '{i}' добавлено: {aug_quantity} аугментаций")
                print(f"Финальный размер трейна: {X_train_with_aug.shape[0]}")
            else:
                print(f"[INFO] Для {self.stage} не заданы аугментации в конфиге: 'augs.{self.stage}'")
        return X_train_with_aug, y_train_with_aug

    def _convert_to_ohe(self, *y_arrays: np.ndarray) -> List[np.ndarray]:
        """
        Применяем to_categorical ко всем переданным массивам меток.
        Используется, если задача мультиклассовая (self.stage in self.multi_stages).
        Иначе позвращаем не измененные массивы.

        :param y_arrays: произвольное число numpy-массивов с метками
        :return: список преобразованных массивов в OHE-формате
        """
        if self.stage not in self.multi_stages:
            return list(y_arrays)

        num_classes=self.num_classes
        return [to_categorical(y, num_classes=num_classes) for y in y_arrays]

    def _normalize_windows(self, X_train:np.ndarray, X_val:np.ndarray, X_test:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Нормализует все окна одним из методов
        
        :param X: numpy array (n_samples, window_size) или (n_samples, window_size, n_channels)
        :return: нормализованный X
        """
        method = config['data']['normalization']        ## 'standard' или 'minmax'

        scalers =[]
        # --- Нормализация ---

        # Если 3-канальное (с производными):
        if len(X_train.shape) == 3:
            _, _, channels = X_train.shape
            for i in range(channels):
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Неизвестный метод нормализации: {method}")

                # Обучаем только на трейне этого канала
                scaler.fit(X_train[..., i].reshape(-1, 1))
                scalers.append(scaler)

            # --- Применяем нормализацию по каждому каналу ---
            X_train_scaled = np.zeros_like(X_train)
            X_val_scaled = np.zeros_like(X_val)
            X_test_scaled = np.zeros_like(X_test)

            for i in range(channels):
                # Для номрмализации меняем форму на (N, 1), после нормализации меняем на обратную по запомненной форме shape
                shape_X_train = X_train[..., i].shape
                shape_X_val = X_val[..., i].shape
                shape_X_test = X_test[..., i].shape
                
                X_train_scaled[..., i] = scalers[i].transform((X_train[..., i].reshape(-1, 1))).reshape(shape_X_train)
                X_val_scaled[..., i] = scalers[i].transform((X_val[..., i].reshape(-1, 1))).reshape(shape_X_val)
                X_test_scaled[..., i] = scalers[i].transform((X_test[..., i].reshape(-1, 1))).reshape(shape_X_test)

        # Если 1-канальное (только оригинальные показания):
        else:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Неизвестный метод нормализации: {method}")

            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            scalers = [scaler]

        # Сохраняем нормализаторы
        save_path = config['paths']['scaler_dir']
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(scalers, f'{save_path}/{self.prefix}_{self.stage}_scalers_{method}.pkl')

        return X_train_scaled, X_val_scaled, X_test_scaled

    def _build_augmentation_table(self, aug_counter: Mapping[str, Mapping[int, int]], all_channels: List[str], all_labels: List[int]):
        
        """
        Формирует данные для таблицы на основе aug_counter
        :param aug_counter: словарь {channel -> label -> count}
        :param all_channels: список всех каналов
        :param all_labels: список всех меток
        """

        # Собираем заголовки
        headers = ["Label"] + all_channels + ["Total"]

        # Формируем таблицу
        table = []

        for label_type in sorted(all_labels):
            row = [label_type]
            total = 0

            for ch in all_channels:
                cnt = aug_counter[ch][label_type]
                row.append(cnt)
                total += cnt

            row.append(total)  # общее число аугментаций по этой метке
            table.append(row)
        # Печатаем красивую таблицу
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def _process_augmentations(
        self,
        x_aug_win: Mapping[str, Mapping[int, List[np.ndarray]]],
        y_aug: Mapping[str, Mapping[int, List[int]]],
        x_aug_total: List[np.ndarray],
        y_aug_total: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Обрабатывает собранные аугментации:
        - Выводит статистику по количеству аугментаций на класс и канал
        - Равномерно выбирает аугментации до лимита `self.size_augmentations`
        - Добавляет их в итоговые списки x_aug_total и y_aug_total

        :param x_aug_win: словарь аугментированных окон (канал → метка → список окон)
        :param y_aug: словарь меток к аугментациям (канал → метка → список меток)
        :param x_aug_total: список, куда будут добавлены выбранные аугментации
        :param y_aug_total: список, куда будут добавлены соответствующие метки
        :return: обновлённые x_aug_total, y_aug_total
        """

        # Все доступные каналы и метки из x_aug_win
        all_channels = list(x_aug_win.keys())
        all_labels = set()
        aug_counter = defaultdict(lambda: defaultdict(int))  # channel -> label_type -> count

        for channel in all_channels:
            all_labels.update(x_aug_win[channel].keys())               # Собираем все используемые в аугментации метки (классы от 0 до N)
        all_labels = sorted(all_labels)                                # Сортируем все метки (классы от 0 до N)

        # Теперь для каждой метки делаем адаптивный сбор
        for label_type in all_labels:
            if label_type == 0:
                continue  # пропускаем нормальные метки

            # Сколько всего каналов содержит эту метку
            channels_with_label = [ch for ch in all_channels if label_type in x_aug_win[ch]]
            k = len(channels_with_label)
            
            ### print(f"\n[Метка: {label_type}] Найдено каналов: {k} → целевой лимит: 20000")

            if not channels_with_label:
                continue

            # Список (канал, количество аугментаций) → чтобы сортировать от меньшего к большему
            channel_counts = [
                (ch, len(x_aug_win[ch][label_type])) 
                for ch in channels_with_label
            ]

            # Сортируем по количеству аугментаций в канале
            channel_counts.sort(key=lambda x: x[1])

            remaining_quota = self.size_augmentations
            used_channels = 0
            total_added_for_label = 0

            # Собираем список аугментаций, начиная с самых маленьких
            for channel, count_in_channel in channel_counts:
                current_k = k - used_channels  # актуальное число нерассмотренных каналов
                current_quota = remaining_quota // max(1, current_k)

                take = min(count_in_channel, current_quota) 
                
                # print(f"Метка {label_type}, канал [{channel}]. Еще надо: {remaining_quota}, на {current_k}, квота {current_quota}. Доступно: {count_in_channel}, берем: {take}")

                samples = x_aug_win[channel][label_type]
                labels = y_aug[channel][label_type]

                # Если по данному каналу есть метки. то собираем их
                if take > 0 and samples:
                    indices = np.random.permutation(len(samples))[:take]
                    selected_samples = [samples[i] for i in indices]
                    selected_labels = np.array(labels)[indices]

                    x_aug_total.extend(selected_samples)
                    y_aug_total.extend(selected_labels)

                    remaining_quota -= take
                    total_added_for_label += take
                    used_channels += 1
    
                    # Добавляем счётчик для каждого (channel, label_type)
                    aug_counter[channel][label_type] += take

        self._build_augmentation_table(aug_counter, all_channels, all_labels)

        return x_aug_total, y_aug_total

    def _save_augmentations_per_labels(self, x_aug_total:list, y_aug_total:list) -> None:
        """
        Сохраняет аугментированные файлы по каждой метке
        """
        save_path = config['paths']['data_dir']
        os.makedirs(save_path, exist_ok=True)
       
        # Переводим в numpy один раз (быстрее и понятнее)
        x_aug_array = np.array(x_aug_total)
        y_aug_array = np.array(y_aug_total)

        # Перебираем метки и по каждой сохраняем список аугментаций в отдельный файл
        unique_labels = np.unique(y_aug_array)
        for lbl in unique_labels:

            # Выбираем по маске только элементы относящиеся к метке
            mask = (y_aug_array == lbl)
            x_lbl = x_aug_array[mask]
            y_lbl = y_aug_array[mask]

            # Перемешиваем список аугментаций
            indices = np.random.permutation(len(x_lbl))
            x_lbl = x_lbl[indices]
            y_lbl = y_lbl[indices]

            filename = f'{save_path}/label_{int(lbl)}_for_{self.stage}.pkl'
            joblib.dump((x_lbl, y_lbl), filename)

    def _save_dataset(self, X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, prefix:str, stage:str) -> None:
        """
        Сохраняет датасет на диск с префиксами
        
        :param prefix: str, например 'top', 'cross', 'uni1', 'uni2', 'total'
        :param stage: str, например 'stage1', 'stage2', 'stage2a', 'stage3'
        """
        dir_to_save = config['paths']['data_dir']
        os.makedirs(dir_to_save, exist_ok=True)

        savefile = os.path.join(dir_to_save, f"{prefix}_{stage}_dataset.npz")
        np.savez(savefile,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )