# dataset_preparing

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from config import config
from typing import Tuple, Dict, Union, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from collections import defaultdict

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
        Основной пайплайн подготовки окон вокруг R-пиков
        
        :param target_channel: str — например, 'MLII'
        :param normalization: str — 'standard' или 'minmax'
        :param add_derivatives: bool — добавлять ли производные
        :return: X_train, y_train, X_val, y_val, X_test, y_test
        """

        for stage in self.all_stages:
            self._check_create_and_save(self.df_top_signals, self.df_top_annotations, self.target_channel_name_1, stage, 'top')
            self._check_create_and_save(self.df_cross_signals, self.df_cross_annotations, self.target_channel_name_2, stage, 'cross')
            self._check_create_and_save(self.df_united_signals_1, self.df_united_annotation_1, self.target_channel_name_1, stage, 'uni1')
            self._check_create_and_save(self.df_united_signals_2, self.df_united_annotation_2, self.target_channel_name_2, stage, 'uni2')
            self._check_create_and_save(self.df_total_signals, self.df_total_annotations, 'Signal', stage, 'total')

    def _check_create_and_save(self, df_signals:pd.DataFrame, df_annotations:pd.DataFrame, channel:str, stage:str, prefix:str):
        """
        Проверяет наличие сохраненного набора датасетов для обучения и валидации модели
        Если не находит: создает и сохраняет (дифференцированно по каждому датасету)
        """
        if not self._is_exsist(stage, prefix):
            X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top = self._create_dataset(df_signals, df_annotations, channel, stage)
            self._save_dataset(X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top, prefix, stage)
            del X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top

    def _is_exsist(self, stage:str, prefix:str) -> bool:
        # Проверка на наличие сохраненного датасета
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

    def _create_dataset(self, df_sign: pd.DataFrame, df_anno: pd.DataFrame, channel: Union[str, Tuple[str]], stage: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.stage = stage
        # формируем окна
        X_windowed, y, X_win_augmented, y_win_augmented = self._create_windows_and_labels(df_sign, df_anno, channel)

        ### print("После _create_windows_and_labels:")
        ### print("X_windowed shape:", X_windowed.shape)
        ### print("y shape:", y.shape)
        ### print("Unique labels in y after windowing:", np.unique(y))
        ### if len(X_windowed) == 0 or len(y) == 0:
        ###     raise ValueError("Сформированные данные пусты. Ошибка на этапе создания окон.")
        if len(y) == 0:
            raise ValueError("Сформированные метки пусты. Ошибка на этапе создания окон.")        

        # Проверка числа классов
        unique_classes = np.unique(y)
        print(f">>> В датасете присутствуют классы: {unique_classes}")
        
        if self.stage in self.multi_stages and len(unique_classes) < 2:
            raise ValueError("На многоклассовой стадии недостаточно уникальных меток для обучения.")
                
        # добавляем производные (если указано)
        if config['data']['add_derivatives']:
            X_augmented = self._add_derivatives_to_windows(X_windowed)
        else:
            X_augmented = X_windowed

        ### print("После _add_derivatives_to_windows:")
        ### print("X_augmented shape:", X_augmented.shape)
        ### if len(X_augmented) == 0:
        ###     raise ValueError("Данные стали пустыми после добавления производных.")

        # разделение на выборки
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_dataset(X_augmented, y)

        # Добавление в обучающую выборку аугментированных образцов
        X_train_with_aug = np.concatenate([X_train, X_win_augmented], axis=0)
        y_train_with_aug = np.concatenate([y_train, y_win_augmented], axis=0)

        print(f"Оригинальных примеров: {X_train.shape[0]}")
        print(f"Аугментаций: {X_win_augmented.shape[0]}")
        print(f"Финальный размер трейна: {X_train_with_aug.shape[0]}")

        ### rint("После _split_dataset:")
        ### rint("X_train shape:", X_train.shape)
        ### rint("X_val shape:", X_val.shape)
        ### rint("X_test shape:", X_test.shape)
        ### rint("y_train unique:", np.unique(y_train))
        ### rint("y_val unique:", np.unique(y_val))
        ### rint("y_test unique:", np.unique(y_test))
        ### f len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        ###    raise ValueError("Одна из выборок оказалась пустой после разбиения.")
        
        # нормализация окон по слоям
        X_train_norm, X_val_norm, X_test_norm = self._normalize_windows(X_train_with_aug, X_val, X_test)

        ### print("После нормализации:")
        ### print("X_train_norm shape:", X_train_norm.shape)
        ### print("X_val_norm shape:", X_val_norm.shape)
        ### print("X_test_norm shape:", X_test_norm.shape)

        print("Выборка сформирована")
        return X_train_norm, y_train_with_aug, X_val_norm, y_val, X_test_norm, y_test

    def _is_label_valid_for_stage(self, row) -> Union[int, None]:
        """Формируем метку в зависимости от стадии"""
        if self.stage == 'stage1_временно_убрано':
            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                return 0  # "Good" (53%)
            else:
                return 1  # "Alert" (47%)

        elif self.stage == 'stage(не реализованная)':
            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                return None         # отсев на 1й стадии
            elif row['Type'] == 'N' and row['Current_Rhythm'] != 'N':
                return 0  # "Attention" (29%)
            else:
                return 1  # "Alarm" (71%)

        elif self.stage == 'stage_all(не реализованная)':
            self.num_classes = 11
            if row['Type'] == 'N':
                return 0  # N: 68.98% == суперкласс Normal
            elif row['Type'] == 'L':
                return 1  # L: 15.25%
            elif row['Type'] == 'R':
                return 2  # R: 13.71%
            elif row['Type'] == 'A':
                return 3  # A: 4.81%
            elif row['Type'] == 'a':
                return 4  # a: 0.28%
            elif row['Type'] == 'J':
                return 5  # J: 0.16%
            elif row['Type'] == 'e':
                return 6  # e: 0.03%
            elif row['Type'] == 'j':
                return 7  # j: 0.43%
            elif row['Type'] == 'V':
                return 8  # V: 13.47%
            elif row['Type'] == 'E':
                return 9  # E: 0.20%
            elif row['Type'] == 'F':
                return 10  # F: 1.52%
            elif row['Type'] == '+':
                return 11  # +: 2.44%
            elif row['Type'] in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 12  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage2a':
            self.num_classes = 11
            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                return None         # отсев на 1й стадии
            elif row['Type'] == 'N' and row['Current_Rhythm'] != 'N':
                return 0  # N: 28.98% == суперкласс Normal

            elif row['Type'] == 'L':
                return 1  # L: 15.25%
            elif row['Type'] == 'R':
                return 2  # R: 13.71%
            elif row['Type'] == 'A':
                return 3  # A: 4.81%
            elif row['Type'] == 'a':
                return 4  # a: 0.28%
            elif row['Type'] == 'J':
                return 5  # J: 0.16%
            elif row['Type'] == 'e':
                return 6  # e: 0.03%
            elif row['Type'] == 'j':
                return 7  # j: 0.43% == суперкласс SVEB

            elif row['Type'] in ['V', 'E']:
                return 8  # V: 13.47%, E: 0.20% == суперкласс VEB: 13.67%

            elif row['Type'] in ['F', '+']:
                return 9  # F: 1.52%, +: 2.44% == суперкласс Fusion: 3.96%

            elif row['Type'] in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 10  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage2':
            self.num_classes = 7
            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                return None         # отсев на 1й стадии
            elif row['Type'] == 'N' and row['Current_Rhythm'] != 'N':
                return 0  # N: 28.98% == суперкласс Normal

            elif row['Type'] == 'L':
                return 1  # L: 15.25% == LBBB
            elif row['Type'] == 'R':
                return 2  # R: 13.71% == RBBB
            elif row['Type'] in ['A', 'a', 'J', 'e', 'j']:
                return 3  # A: 4.81%, a: 0.28%, J: 0.16%, e: 0.03%, j 0.43% == суперкласс subSVEB

            elif row['Type'] in ['V', 'E']:
                return 4  # V: 13.47%, E: 0.20% == суперкласс VEB: 13.67%

            elif row['Type'] in ['F', '+']:
                return 5  # F: 1.52%, +: 2.44% == суперкласс Fusion: 3.96%

            elif row['Type'] in ['Q', '/', '!', '~', 'f', 'U', '?', '"', 'x', '[', ']']:
                return 6  # шумы: 18.45% == суперкласс Q
            else:
                return None

        elif self.stage == 'stage3':
            self.num_classes = 5

            if row['Type'] in ['A']:
                return 0  # A: 4.81%
            elif row['Type'] in ['a']:
                return 1  # a: 0.28%
            elif row['Type'] in ['J']:
                return 2  # J: 0.16%
            elif row['Type'] in ['e']:
                return 3  # e: 0.03%
            elif row['Type'] in ['j']:
                return 4  # j 0.43%
            else:
                return None

        elif self.stage == 'stage1':
            self.num_classes = 24
            if row['Type'] == 'N':
                return 0  # N: 68.98% == суперкласс Normal
            elif row['Type'] == 'L':
                return 1  # L: 15.25%
            elif row['Type'] == 'R':
                return 2  # R: 13.71%
            elif row['Type'] == 'A':
                return 3  # A: 4.81%
            elif row['Type'] == 'a':
                return 4  # a: 0.28%
            elif row['Type'] == 'J':
                return 5  # J: 0.16%
            elif row['Type'] == 'e':
                return 6  # e: 0.03%
            elif row['Type'] == 'j':
                return 7  # j: 0.43%
            elif row['Type'] == 'V':
                return 8  # V: 13.47%
            elif row['Type'] == 'E':
                return 9  # E: 0.20%
            elif row['Type'] == 'F':
                return 10  # F: 1.52%
            elif row['Type'] == '+':
                return 11  # +: 2.44%
            elif row['Type'] == 'Q':
                return 12  # шумы: 
            elif row['Type'] == '/':
                return 13  # шумы: 
            elif row['Type'] == '!':
                return 14  # шумы: 
            elif row['Type'] == '~':
                return 15  # шумы: 
            elif row['Type'] == 'f':
                return 16  # шумы: 
            elif row['Type'] == 'U':
                return 17  # шумы: 
            elif row['Type'] == '?':
                return 18  # шумы: 
            elif row['Type'] == '"':
                return 19  # шумы: 
            elif row['Type'] == 'x':
                return 20  # шумы: 
            elif row['Type'] == '[':
                return 21  # шумы: 
            elif row['Type'] == ']':
                return 22  # шумы: 
            elif row['Type'] == '|':
                return 23  # шумы: 
            else:
                return None
            
    def _create_windows_and_labels(self,
        df_signals: pd.DataFrame,
        df_annotations: pd.DataFrame,
        channels: Union[str, Tuple[str]]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        ### print("Входные данные в _create_windows_and_labels:")
        ### print(f"stage: {self.stage}")
        ### print("df_signals shape:", df_signals.shape)
        ### print("df_annotations shape:", df_annotations.shape)
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

                ### print(f"Обрабатываем Patient_id: {pid}")
                ### print(f"df_p_signal shape: {df_p_signal.shape}, df_p_signal колонки: {df_p_signal.columns}")
                ### print(f"df_p_annotation shape: {df_p_annotation.shape}, df_p_annotation колонки: {df_p_annotation.columns}")


                ### # Для каждой аннотации у этого пациента
                ### for _, row in df_p_annotation.iterrows():
                ###     sample = row['Sample']
                ###     start = sample - half_window
                ###     end = sample + half_window
                ### 
                ###     start_canvas = start - half_shift_size
                ###     end_canvas = end + half_shift_size
                ### 
                ###     # if channel_filter == 'total':
                ###     if 'Channel' in row:
                ###         channel_type = row['Channel'] # 
                ###         # Извлекаем участок сигнала
                ###         window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end) & (df_p_signal['Channel'] == channel_type)]
                ###         # Извлекаем участок для формирования ряда аугментированных шифтингом сигналов
                ###         aug_canvas = df_p_signal[(df_p_signal['Sample'] >= start_canvas) & (df_p_signal['Sample'] < end_canvas) & (df_p_signal['Channel'] == channel_type)]
                ### 
                ###     else:
                ###         # Извлекаем участок сигнала
                ###         window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end)]
                ###         aug_canvas = df_p_signal[(df_p_signal['Sample'] >= start_canvas) & (df_p_signal['Sample'] < end_canvas)]

                    ### print(f"[DEBUG] Sample: {sample}, start: {start}, end: {end}")
                    ### print(f"[DEBUG] window shape: {window.shape}")

                    # Избавляемся от неполных окон по краям набора данных
                    if len(window) != window_size:
                        ### print(f">>> [DEBUG] len(window): {len(window)}, window_size: {window_size}")
                        continue


                    ###############################################################
                    # ФОРМИРОВАНИЕ НАБОРА ( LABELS ) 
                    # Проверяем валидность метки для стадии
                    ###############################################################
                    label = self._is_label_valid_for_stage(row)

                    ### print("Входные данные в _create_windows_and_labels:")
                    ### print(f"stage: {self.stage}")
                    ### print("df_signals shape:", df_signals.shape)
                    ### print("df_annotations shape:", df_annotations.shape)
                    ### print(f"target_channel: {target_channel} в {channels}")
                    ### print(f"[DEBUG out] Type: {row['Type']}, Rhythm: {row.get('Current_Rhythm', None)}, Label: {label}")

                    if label is None:
                        print(f"[DEBUG in] Type: {row['Type']}, Rhythm: {row.get('Current_Rhythm', None)}, Label: {label}")
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
                        if len(x_aug_win[channel_type][label_type]) < 10000:
                            print(f"Для метки [{label_type}] по каналу [{channel_type}] аугментировано {len(x_aug_win[channel_type][label_type])} копий. Начал аугментацию для {sample}...")
                            aug_signal_values = self._shift_noise_augmentation(
                                np.array(aug_canvas[target_channel].values),
                                np.array(signal_values)
                                )
                            x_aug_win[channel_type][label_type].extend(aug_signal_values)
                            y_aug[channel_type][label_type].extend([label] * len(aug_signal_values))      ## добавляем список меток по числу аугментированных копий
                            print(f"Закончил аугментацию для {sample}. Для метки [{label_type}] аугментировано {len(x_aug_win[channel_type][label_type])} копий.")
                        ### else:
                        ###     print(f"len(x_aug_win[channel_type][label_type]) >= 10000 {len(x_aug_win[channel_type][label_type])}")

                    ### else:
                    ###     print(f"[DEBUG in] label_type == 'N': {label_type}")

                    ###############################################################
                    
                    y.append(label)
                    x_win.append(signal_values)

        for channel in x_aug_win:
            for label_type in x_aug_win[channel]:
                # Окна выбранного типа (метка) по данному каналу
                aug_samples = x_aug_win[channel][label_type]
                # Метки для окон
                aug_labels = y_aug[channel][label_type]
                if not aug_samples:
                    print(f"В канале [{channel}] найдено {len(aug_samples)} меток [{label_type}].")
                    continue
                
                # Считаем количество каналов содержащих данную метку
                k = sum(1 for chan in x_aug_win if label_type in x_aug_win[chan])
                # Минимальная доля копий на каждый канал
                per_channel_limit = 20000 // k

                # Берём не больше, чем положено на канал
                take = min(per_channel_limit, len(aug_samples))

                # Перемешиваем выборку данному типа (метка) аугментаций по данному каналу
                indices_local = np.random.permutation(len(aug_samples))
                aug_samples = [aug_samples[i] for i in indices_local]
                aug_labels = np.array(aug_labels)[indices_local]

                # Добавляем долю даннного канала в общую выборку по метке
                x_aug_total.extend(aug_samples[:take])
                y_aug_total.extend(aug_labels[:take])

                print(f"Добавлено {take} аугментаций: канал [{channel}], метка [{label_type}]")

        # Перемешиваем финальный список аугментаций
        indices = np.random.permutation(len(x_aug_total))
        x_aug_total = [x_aug_total[i] for i in indices]
        y_aug_total = np.array(y_aug_total)[indices]

        if len(x_win) == 0:
            raise ValueError("Не удалось создать ни одного окна. Возможно, неверный размер окна или фильтрация слишком жёсткая.")
        if len(np.unique(y)) < 2:
            raise ValueError("Недостаточно уникальных меток для обучения модели.")

        np_y = np.array(y)
        np_y_aug = np.array(y_aug_total)

        # Если мультикласс, то метки переводим в ohe-формат
        if self.stage in self.multi_stages:
            np_y = to_categorical(np_y, num_classes=self.num_classes)
            np_y_aug = to_categorical(np_y_aug, num_classes=self.num_classes)

        print(f">> Всего создано {len(x_win)} окон, и дополнительно {len(x_aug_total)} аугментированных окон.")
        print(f">> В списке 'y' {len(np.unique(y))} меток: [{np.unique(y)}].")
        print(f">> В аугм. списке 'y_aug' {len(np.unique(y_aug_total))} меток: [{np.unique(y_aug_total)}].")

        return np.array(x_win), np_y, np.array(x_aug_total), np_y_aug

    def _create_windows_and_labels_old(self,
        df_signals: pd.DataFrame,
        df_annotations: pd.DataFrame,
        channels: Union[str, Tuple[str]]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        x_aug_win = defaultdict(list)
        y_aug = defaultdict(list)
        x_aug_total = []
        y_aug_total = []

        ### print("Входные данные в _create_windows_and_labels:")
        ### print(f"stage: {self.stage}")
        ### print("df_signals shape:", df_signals.shape)
        ### print("df_annotations shape:", df_annotations.shape)
        for target_channel in channels:
            ### print(f"target_channel: {target_channel} в {channels}")
    
            for pid in tqdm(df_annotations['Patient_id'].unique(), desc="Формируем окна"):
                # Выбираем данные пациента
                df_p_signal = df_signals[df_signals['Patient_id'] == pid]
                df_p_annotation = df_annotations[df_annotations['Patient_id'] == pid]

                ### print(f"Обрабатываем Patient_id: {pid}")
                ### print(f"df_p_signal shape: {df_p_signal.shape}, df_p_signal колонки: {df_p_signal.columns}")
                ### print(f"df_p_annotation shape: {df_p_annotation.shape}, df_p_annotation колонки: {df_p_annotation.columns}")

                # Для каждой аннотации у этого пациента
                for _, row in df_p_annotation.iterrows():
                    sample = row['Sample']
                    start = sample - half_window
                    end = sample + half_window

                    start_canvas = start - half_shift_size
                    end_canvas = end + half_shift_size

                    # if channel_filter == 'total':
                    if 'Channel' in row:
                        channel_type = row['Channel'] # 
                        # Извлекаем участок сигнала
                        window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end) & (df_p_signal['Channel'] == channel_type)]
                        # Извлекаем участок для формирования ряда аугментированных шифтингом сигналов
                        aug_canvas = df_p_signal[(df_p_signal['Sample'] >= start_canvas) & (df_p_signal['Sample'] < end_canvas) & (df_p_signal['Channel'] == channel_type)]

                    else:
                        # Извлекаем участок сигнала
                        window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end)]
                        aug_canvas = df_p_signal[(df_p_signal['Sample'] >= start_canvas) & (df_p_signal['Sample'] < end_canvas)]

                    ### print(f"[DEBUG] Sample: {sample}, start: {start}, end: {end}")
                    ### print(f"[DEBUG] window shape: {window.shape}")

                    # Избавляемся от неполных окон по краям набора данных
                    if len(window) != window_size:
                        continue

                    ### print(f">>> [DEBUG] len(window): {len(window)}, window_size: {window_size}")

                    ###############################################################
                    # ФОРМИРОВАНИЕ НАБОРА ( LABELS ) 
                    # Проверяем валидность метки для стадии
                    ###############################################################
                    label = self._is_label_valid_for_stage(row)

                    ### print("Входные данные в _create_windows_and_labels:")
                    ### print(f"stage: {self.stage}")
                    ### print("df_signals shape:", df_signals.shape)
                    ### print("df_annotations shape:", df_annotations.shape)
                    ### print(f"target_channel: {target_channel} в {channels}")
                    ### print(f"[DEBUG out] Type: {row['Type']}, Rhythm: {row.get('Current_Rhythm', None)}, Label: {label}")

                    if label is None:
                        ### print(f"[DEBUG in] Type: {row['Type']}, Rhythm: {row.get('Current_Rhythm', None)}, Label: {label}")
                        continue

                    signal_values = window[target_channel].values

                    ###############################################################
                    # АУГМЕНТАЦИЯ
                    # Полуение списка аугментированных копий для signal_values
                    ###############################################################
                    label_type = row['Type']
                    if label_type != 'N':
                        if sample // 100000:
                            print(f"Проверяем {sample}...")

                        if len(x_aug_win[label_type]) < 20000:
                            print(f"Для метки [{label_type}] аугментировано {len(x_aug_win[label_type])} копий. Начал аугментацию для {sample}...")
                            aug_signal_values = self._shift_noise_augmentation(
                                np.array(aug_canvas[target_channel].values),
                                np.array(signal_values)
                                )
                            x_aug_win[label_type].extend(aug_signal_values)
                            y_aug[label_type].extend([label] * len(aug_signal_values))      ## добавляем список меток по числу аугментированных копий
                            print(f"Закончил аугментацию для {sample}. Для метки [{label_type}] аугментировано {len(x_aug_win[label_type])} копий.")
                    ###############################################################
                    
                    y.append(label)
                    x_win.append(signal_values)

        for key, value in x_aug_win.items():
            x_aug_total.extend(value)
            print(f"Для '{key}' создано {len(value)} аугментироанных окон")
        for key, value in y_aug.items():
            y_aug_total.extend(value)

        ### print(f"Unique labels in y after filtering: {np.unique(y)}")                    

        if len(x_win) == 0:
            raise ValueError("Не удалось создать ни одного окна. Возможно, неверный размер окна или фильтрация слишком жёсткая.")
        if len(np.unique(y)) < 2:
            raise ValueError("Недостаточно уникальных меток для обучения модели.")

        np_y = np.array(y)
        np_y_aug = np.array(y_aug_total)

        # Если мультикласс, то метки переводим в ohe-формат
        if self.stage in self.multi_stages:
            np_y = to_categorical(np_y, num_classes=self.num_classes)
            np_y_aug = to_categorical(np_y_aug, num_classes=self.num_classes)

        print(f">> Всего создано {len(x_win)} окон, и дополнительно {sum(len(val) for val in x_aug_win.values())} аугментированных окон.")
        print(f">> В списке 'y' {len(np.unique(y))} меток: [{np.unique(y)}].")
        print(f">> В аугм. списке 'y_aug' {len(np.unique(y_aug_total))} меток: [{np.unique(y_aug_total)}].")

        print(f">> В списке 'np_y' {len(np_y[np_y == 0])} 'N' меток.")
        for i in range(len(np.unique(y))):
            print(f">> В аугм. списке 'y_aug' {len(np_y_aug[np_y_aug == i])} меток: '{i}'.")

        return np.array(x_win), np_y, np.array(x_aug_total), np_y_aug

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

        X_with_derivatives = []

        for signal in X:
            d1 = np.gradient(signal)
            d2 = np.gradient(d1)
            combined = np.stack([signal, d1, d2], axis=-1)  # (window_size, 3)
            X_with_derivatives.append(combined)

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

            ### print(f"X_train.shape перед нормализацией: {X_train.shape}")
            ### print(f"X_val.shape перед нормализацией: {X_val.shape}")
            ### print(f"X_test.shape перед нормализацией: {X_test.shape}")

            # --- Применяем нормализацию по каждому каналу ---
            X_train_scaled = np.zeros_like(X_train)
            X_val_scaled = np.zeros_like(X_val)
            X_test_scaled = np.zeros_like(X_test)

            for i in range(channels):
                #X_train_scaled[..., i] = scalers[i].transform(X_train[..., i].reshape(-1, 1)).flatten()
                #X_val_scaled[..., i] = scalers[i].transform(X_val[..., i].reshape(-1, 1)).flatten()
                #X_test_scaled[..., i] = scalers[i].transform(X_test[..., i].reshape(-1, 1)).flatten()

                ### print(f"Канал {i}. До Norm => X_train[..., {i}].shape = {X_train[..., i].shape}")
                ### print(f"Канал {i}. До Norm => X_val[..., {i}].shape = {X_val[..., i].shape}")
                ### print(f"Канал {i}. До Norm => X_test[..., {i}].shape = {X_test[..., i].shape}")

                # Для номрмализации меняем форму на (N, 1), после нормализации меняем на обратную по запомненной форме shape
                shape_X_train = X_train[..., i].shape
                shape_X_val = X_val[..., i].shape
                shape_X_test = X_test[..., i].shape

                ### print(f"Форма shape_X_train = {shape_X_train}")
                ### print(f"Форма shape_X_val = {shape_X_val}")
                ### print(f"Форма shape_X_test = {shape_X_test}")
                
                X_train_scaled[..., i] = scalers[i].transform((X_train[..., i].reshape(-1, 1))).reshape(shape_X_train)
                X_val_scaled[..., i] = scalers[i].transform((X_val[..., i].reshape(-1, 1))).reshape(shape_X_val)
                X_test_scaled[..., i] = scalers[i].transform((X_test[..., i].reshape(-1, 1))).reshape(shape_X_test)

                ### print(f"Канал {i}. После Norm => X_train_scaled[..., {i}].shape = {X_train_scaled[..., i].shape}")
                ### print(f"Канал {i}. После Norm => X_val_scaled[..., {i}].shape = {X_val_scaled[..., i].shape}")
                ### print(f"Канал {i}. После Norm => X_test_scaled[..., {i}].shape = {X_test_scaled[..., i].shape}")

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
        return X_train_scaled, X_val_scaled, X_test_scaled

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