# src/dataset_filtering.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import Tuple

class DatasetFiltering:
    def __init__(self, data_container):
        """
        Инициализирует фильтр, получая доступ к контейнеру данных.
        """
        self.container = data_container

    def run(self) -> None:
        """
        Выполняет полный цикл фильтрации данных.
        """
        self._check_channels_per_patient()
        self._analyze_channels_statistics()
        self._add_rhythm_annotations()
        self._filter_dataframes()

    def _check_channels_per_patient(self) -> None:
        """
        Проверяет, какие каналы доступны у каждого пациента.
        Формирует self.channels_per_patient — словарь:
            {pid: {'channels': [список], 'ohe': [вектор]}}
        """
        all_channels = ['MLII', 'V1', 'V2', 'V4', 'V5']

        for pid in tqdm(self.container.patient_ids, desc="Проверяем каналы", unit=" пациент"):
            df_p = self.container.df_all_signals[self.container.df_all_signals['Patient_id'] == pid]
            available_channels = [
                ch for ch in all_channels if ch in df_p.columns and not df_p[ch].isna().all()
            ]
            ohe_vector = [1 if ch in available_channels else 0 for ch in all_channels]
            self.container.channels_per_patient[pid] = {
                'channels': available_channels,
                'ohe': ohe_vector
            }

    def _analyze_channels_statistics(self) -> None:
        """
        Вспомогательный метод. Нужен для анализа данных и принятия решения о выборе каналов,
        на основе которых будет формироваться однородные данные будущего ДС.
        """

        channel_counter = Counter()
        for data in self.container.channels_per_patient.values():
            channel_counter.update(data['channels'])
        # === Таблица распределения каналов по пациентам ===
        print("\n=== Таблица распределения каналов по пациентам ===")
        print("Канал | Число пациентов с этим каналом")
        print("-" * 35)

        for ch, count in sorted(channel_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{ch.ljust(6)}| {count}")
        
        # === Автоматический выбор двух самых популярных каналов ===
        total_patients = int(len(self.container.patient_ids))
        # Получаем список каналов, отсортированных по числу пациентов (в порядке убывания)
        sorted_channels = sorted(channel_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Берём 1-й по популярности канал
        self.container.target_channel_name_1 = sorted_channels[0][0]
        channel_1_count = sorted_channels[0][1]
        channel_1_percentage = (channel_1_count / total_patients) * 100

        # Берём 2-й по популярности канал
        self.container.target_channel_name_2 = sorted_channels[1][0]
        channel_2_count = sorted_channels[1][1]
        channel_2_percentage = (channel_2_count / total_patients) * 100

        # === Вывод результатов ===
        print("\n=== Лидирующие каналы ===")
        print(f"Наиболее распространённый канал: {self.container.target_channel_name_1}")
        print(f"Число пациентов: {channel_1_count} из {total_patients}")
        print(f"Процент от всех пациентов: {channel_1_percentage:.2f}%")

        print(f"Второй по распространённости канал: {self.container.target_channel_name_2}")
        print(f"Число пациентов: {channel_2_count} из {total_patients}")
        print(f"Процент от всех пациентов: {channel_2_percentage:.2f}%")

        print(f"\nВЫВОД:\nОчевидно, что канал: {self.container.target_channel_name_1} будет выбран в качестве источника сигналов, формирующих датасет.\nЕго показания есть у {channel_1_percentage:.2f}% пациентов.\nДанными оставшихся {total_patients - channel_1_count} ({total_patients} - {channel_1_count}) пациентов можно пожертвовать ради однородности и чистоты датасета.")
        print(f"\nКанал {self.container.target_channel_name_2} можно выбирать для сравнения\nЕго показания есть у {channel_2_percentage:.2f}% пациентов.\nПри сравнении придется пожертвовать данными уже {total_patients - channel_2_count} ({total_patients} - {channel_2_count}) пациентов.")

    def _add_rhythm_annotations(self) -> None:
        """
        Добавляет колонку 'Current_Rhythm' в self.container.df_all_annotations.
        Значение распространяется до следующего изменения ритма.
        """

        # Извлекаем только строки с ритмическими аннотациями
        rhythm_mask = self.container.df_all_annotations['Aux'].notna() & self.container.df_all_annotations['Aux'].str.startswith('(')
        rhythm_df = self.container.df_all_annotations.loc[rhythm_mask, ['Sample', 'Patient_id', 'Aux']].copy()

        # Убираем скобку '(' из значения
        rhythm_df['Current_Rhythm'] = rhythm_df['Aux'].str[1:]  # например, '(AFIB' → 'AFIB'

        # Удаляем исходную колонку Aux (если не нужна далее)
        rhythm_df.drop(columns=['Aux'], inplace=True)

        # Объединяем обратно по Patient_id и Sample
        # НО нам нужно заполнить NaN вниз — forward fill по каждому пациенту
        self.container.df_all_annotations = pd.merge(
            self.container.df_all_annotations,
            rhythm_df[['Sample', 'Patient_id', 'Current_Rhythm']],
            on=['Sample', 'Patient_id'],
            how='left'
        )

        # Заполняем пропуски в Current_Rhythm
        # Сортируем по Patient_id и Sample, чтобы заполнение шло по времени
        self.container.df_all_annotations.sort_values(['Patient_id', 'Sample'], inplace=True)
        self.container.df_all_annotations['Current_Rhythm'] = (
            self.container.df_all_annotations.groupby('Patient_id', group_keys=False)['Current_Rhythm'].apply(lambda x: x.ffill())
        )

        #print("Колонка 'Current_Rhythm' добавлена к аннотациям.")

    def _filter_dataframes(self) -> None:
        """
        Формирует датафреймы сигналов и аннотаций (подходящие для любой стадии) по двум каналам:
            - self.container.df_top_signals / self.container.df_top_annotations → обучение на основном канале (target_channel_name_1)
            - self.container.df_cross_signals / self.container.df_cross_annotations → тестирование на втором канале (target_channel_name_2)
            - self.container.df_united_signals_1 / self.container.df_united_annotation_1 → данные для модели 1
            - self.container.df_united_signals_2 / self.container.df_united_annotation_2 → данные для модели 2
            self.container.df_full_signals
            self.container.df_full_annotation
        """

        # Фильтруем по первому каналу
        self.container.df_top_signals, self.container.df_top_annotations = self._filter_df(
            self.container.df_all_signals,
            self.container.df_all_annotations,
            self.container.target_channel_name_1
        )
        print(f"Оставлено записей после фильтрации (по {self.container.target_channel_name_1}): {len(self.container.df_top_signals)}")
        print(f"Оставлено аннотаций после фильтрации (по {self.container.target_channel_name_1}): {len(self.container.df_top_annotations)}")

        # Фильтруем по второму каналу
        self.container.df_cross_signals, self.container.df_cross_annotations = self._filter_df(
            self.container.df_all_signals,
            self.container.df_all_annotations,
            self.container.target_channel_name_2
        )
        print(f"Оставлено записей после фильтрации (по {self.container.target_channel_name_2}): {len(self.container.df_cross_signals)}")
        print(f"Оставлено аннотаций после фильтрации (по {self.container.target_channel_name_2}): {len(self.container.df_cross_annotations)}")

        # Создание ДФ по всем каналам, но выстроенным в один столбец.
        list_of_signals = []
        list_of_annotations = []
        for channel in ['MLII', 'V1', 'V2', 'V4', 'V5']:
            df_sig, df_ann = self._filter_df(
                self.container.df_all_signals,
                self.container.df_all_annotations,
                channel
            )
            # Переименовываем колонку с сигналом в общее имя 'Signal' в ДФ сигналов
            df_sig.rename(columns={channel: 'Signal'}, inplace=True)
            df_ann['Channel'] = channel  # добавляем метку о канале в ДФ сигналов
            df_sig['Channel'] = channel  # добавляем метку о канале в ДФ аннотаций
    
            list_of_signals.append(df_sig)
            list_of_annotations.append(df_ann)            
        self.container.df_total_signals = pd.concat(list_of_signals, ignore_index=True)
        self.container.df_total_annotations = pd.concat(list_of_annotations, ignore_index=True)

        # Находим общих пациентов
        common_pids = np.intersect1d(
            self.container.df_top_signals['Patient_id'].unique(),
            self.container.df_cross_signals['Patient_id'].unique()
        )

        # Двойная фильтрация по первому каналу
        self.container.df_united_signals_1 = self.container.df_top_signals[self.container.df_top_signals['Patient_id'].isin(common_pids)]
        self.container.df_united_annotation_1 = self.container.df_top_annotations[self.container.df_top_annotations['Patient_id'].isin(common_pids)]

        # Двойная фильтрация по второму каналу
        self.container.df_united_signals_2 =  self.container.df_cross_signals[self.container.df_cross_signals['Patient_id'].isin(common_pids)]
        self.container.df_united_annotation_2 =  self.container.df_cross_annotations[self.container.df_cross_annotations['Patient_id'].isin(common_pids)]
        print(f"Оставлено записей после 2-ной фильтрации (по {self.container.target_channel_name_1} и {self.container.target_channel_name_2}): {len(self.container.df_united_signals_1)}")
        print(f"Оставлено аннотаций после 2-ной фильтрации (по {self.container.target_channel_name_1} и {self.container.target_channel_name_2}): {len(self.container.df_united_annotation_1)}")

    def _filter_df(self, df_s, df_a, channel) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Фильтрует сигналы и аннотации по наличию данных в указанном канале.

        :param df_s: исходный датафрейм сигналов (pd.DataFrame)
        :param df_a: исходный датафрейм аннотаций (pd.DataFrame)
        :param channel: название канала для фильтрации (например 'MLII', 'V1')
        :return: df_filtered_s, df_filtered_a
        """
        # Проверяем, существует ли такой канал в сигналах
        if channel not in df_s.columns:
            raise ValueError(f"Канал '{channel}' отсутствует в df_signals")        

        # Фильтруем сигналы по наличию данных в целевом канале
        df_filtered_s = df_s[df_s[channel].notna()][['Sample', channel, 'Patient_id']].copy()

        # Получаем уникальные pid из отфильтрованных сигналов
        valid_pids = df_filtered_s['Patient_id'].unique()

        # Фильтруем аннотации
        df_filtered_a = df_a[df_a['Patient_id'].isin(valid_pids)].copy()

        return df_filtered_s, df_filtered_a