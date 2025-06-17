# dataset_preparing

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from config import config
from typing import Tuple, Dict, Union, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        df_united_annotation_2: pd.DataFrame) -> None:

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

    def pipeline(self) -> Dict:
        """
        Основной пайплайн подготовки окон вокруг R-пиков
        
        :param target_channel: str — например, 'MLII'
        :param normalization: str — 'standard' или 'minmax'
        :param add_derivatives: bool — добавлять ли производные
        :return: X_train, y_train, X_val, y_val, X_test, y_test
        """
        self.ds = {}

        X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top = self._create_dataset(self.df_top_signals, self.df_top_annotations, self.target_channel_name_1)
        self.ds['top'] = self._save_dataset(X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top, 'top', 'stage1')
        del X_train_top, y_train_top, X_val_top, y_val_top, X_test_top, y_test_top
        
        X_train_cross, y_train_cross, X_val_cross, y_val_cross, X_test_cross, y_test_cross = self._create_dataset(self.df_cross_signals, self.df_cross_annotations, self.target_channel_name_2)
        self.ds['cross'] = self._save_dataset(X_train_cross, y_train_cross, X_val_cross, y_val_cross, X_test_cross, y_test_cross, 'cross', 'stage1')
        del X_train_cross, y_train_cross, X_val_cross, y_val_cross, X_test_cross, y_test_cross

        X_train_uni_1, y_train_uni_1, X_val_uni_1, y_val_uni_1, X_test_uni_1, y_test_uni_1 = self._create_dataset(self.df_united_signals_1, self.df_united_annotation_1, self.target_channel_name_1)
        self.ds['uni_1'] = self._save_dataset(X_train_uni_1, y_train_uni_1, X_val_uni_1, y_val_uni_1, X_test_uni_1, y_test_uni_1, 'uni1', 'stage1')
        del X_train_uni_1, y_train_uni_1, X_val_uni_1, y_val_uni_1, X_test_uni_1, y_test_uni_1

        X_train_uni_2, y_train_uni_2, X_val_uni_2, y_val_uni_2, X_test_uni_2, y_test_uni_2 = self._create_dataset(self.df_united_signals_2, self.df_united_annotation_2, self.target_channel_name_2)
        self.ds['uni_2'] = self._save_dataset(X_train_uni_2, y_train_uni_2, X_val_uni_2, y_val_uni_2, X_test_uni_2, y_test_uni_2, 'uni2', 'stage1')
        del X_train_uni_2, y_train_uni_2, X_val_uni_2, y_val_uni_2, X_test_uni_2, y_test_uni_2

        return self.ds

    def _create_dataset(self, df_sign: pd.DataFrame, df_anno: pd.DataFrame, channel: str) -> Tuple:
        # формируем окна
        X_windowed, y = self._create_windows_and_labels(df_sign, df_anno, channel)

        ### print("После _create_windows_and_labels:")
        ### print("X_windowed shape:", X_windowed.shape)
        ### print("y shape:", y.shape)
        ### print("Unique labels in y after windowing:", np.unique(y))
        ### if len(X_windowed) == 0 or len(y) == 0:
        ###     raise ValueError("Сформированные данные пусты. Ошибка на этапе создания окон.")
        
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
        X_train_norm, X_val_norm, X_test_norm = self._normalize_windows(X_train, X_val, X_test)

        ### print("После нормализации:")
        ### print("X_train_norm shape:", X_train_norm.shape)
        ### print("X_val_norm shape:", X_val_norm.shape)
        ### print("X_test_norm shape:", X_test_norm.shape)

        print("Выборка сформирована")
        return X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test

    def _create_windows_and_labels(self,
        df_signals: pd.DataFrame,
        df_annotations: pd.DataFrame,
        channels: Union[str, Tuple[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формирует окна вокруг R-пиков
        
        :param df_signals: pd.DataFrame с Sample и каналом target_channel
        :param df_annotations: pd.DataFrame с Sample и метками Type + Current_Rhythm
        :param target_channel: str — имя канала, например 'MLII'
        :return: numpy array (X), numpy array (y)
        """

        window_size = config['data']['window_size']  # например, 360
        half_window = window_size // 2

        x_win = []
        y = []

        ### print("Входные данные в _create_windows_and_labels:")
        ### print("df_signals shape:", df_signals.shape)
        ### print("df_annotations shape:", df_annotations.shape)
        ### print("target_channel:", target_channel)
        for target_channel in channels:
            for pid in tqdm(df_annotations['Patient_id'].unique(), desc="Формируем окна"):
                # Выбираем данные пациента
                df_p_signal = df_signals[df_signals['Patient_id'] == pid]
                df_p_annotation = df_annotations[df_annotations['Patient_id'] == pid]

                ### print(f"Обрабатываем Patient_id: {pid}")
                ### print("df_p_signal shape:", df_p_signal.shape)
                ### print("df_p_annotation shape:", df_p_annotation.shape)

                # Для каждой аннотации у этого пациента
                for _, row in df_p_annotation.iterrows():
                    ### print(f">>>> row = {row}\n")
                    sample = row['Sample']
                    start = sample - half_window
                    end = sample + half_window

                    # Извлекаем участок сигнала
                    window = df_p_signal[(df_p_signal['Sample'] >= start) & (df_p_signal['Sample'] < end)]
                    ### print(f">>>> len(window) = {len(window)}")
                    
                    # Избавляемся от неполных окон по краям набора данных
                    if len(window) != window_size:
                        continue

                    signal_values = window[target_channel].values
                    x_win.append(signal_values)
                    ### print(f">>>> len(x_win) = {len(x_win)}")
                    # Формируем метку
                    if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                        y.append(0)  # "Good"
                    else:
                        y.append(1)  # "Alert"

        ### print("Количество сформированных окон:", len(x_win))
        ### print("Unique меток в y:", np.unique(y))
        if len(x_win) == 0:
            raise ValueError("Не удалось создать ни одного окна. Возможно, неверный размер окна или фильтрация слишком жёсткая.")
        if len(np.unique(y)) < 2:
            raise ValueError("Недостаточно уникальных меток для обучения модели.")

        return np.array(x_win), np.array(y)
    
    def _add_derivatives_to_windows(self, X):
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

    def _split_dataset(self, X, y):
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

    def _normalize_windows(self, X_train, X_val, X_test):
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

    def _save_dataset(self, X_train, y_train, X_val, y_val, X_test, y_test, prefix, stage):
        """
        Сохраняет датасет на диск с префиксами
        
        :param prefix: str, например 'top', 'cross', 'uni1', 'uni2'
        :param stage: str, например 'stage1', 'stage2'            
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
        return savefile