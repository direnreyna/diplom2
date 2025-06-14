# dataset_analyzer.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import config
from typing import Tuple, List
from collections import Counter

class DatasetAnalyze:
    def __init__(self, df_all_signals: pd.DataFrame, df_all_annotations: pd.DataFrame, patient_ids: List) -> None:
        self.temp_dir = config['paths']['temp_dir']
        self.check_analytics = config['data']['analytics']
        
        self.patient_ids = patient_ids                          # Список всех ID пациентов (берётся из имён файлов)
        self.df_all_signals = df_all_signals                    # Все сигналы ЭКГ до фильтрации: по всем каналам
        self.df_all_annotations = df_all_annotations            # Все аннотации
        
        self.channels_per_patient = {}                          # Информация по каждому пациенту: какие каналы доступны, OHE-вектор
        self.target_channel_name_1 = ''                         # Имя основного канала для обучения (например, 'MLII')
        self.target_channel_name_2 = ''                         # Имя второго канала для сравнения (например, 'V1')
        
        self.df_top_signals = pd.DataFrame()                    # Отфильтрованные сигналы для формирования ДС по 1 топ-каналу
        self.df_top_annotations = pd.DataFrame()                # Аннотации для формирования ДС по 1 топ-каналу
        self.df_cross_signals = pd.DataFrame()                  # Отфильтрованные сигналы для формирования кросс-теста по не топ-каналу
        self.df_cross_annotations = pd.DataFrame()              # Аннотации для формирования кросс-теста по не топ-каналу

        self.df_united_signals_1 = pd.DataFrame()               # Отфильтрованные сигналы для формирования 1го ДС по 2 топ-каналам
        self.df_united_annotation_1 = pd.DataFrame()            # Аннотации для формирования 1го ДС по 2 топ-каналам
        self.df_united_signals_2 = pd.DataFrame()               # Отфильтрованные сигналы для формирования 2го ДС по 2 топ-каналам
        self.df_united_annotation_2 = pd.DataFrame()            # Аннотации для формирования 2го ДС по 2 топ-каналам

        self.df_global_peak_distribution = pd.DataFrame()       # Общее распределение типов R-пиков: Count, Percent
        self.df_patient_peak_top_types = pd.DataFrame()         # Топ-5 типов пиков на пациента → только проценты
        self.df_patient_normal_abnormal_ratio = pd.DataFrame()  # Процент нормальных / аномальных пиков на пациента
        self.df_patient_top_anomaly = pd.DataFrame()            # Самый частый аномальный пик у каждого пациента + его доля
        self.rhythm_types = {}                                  # Словарь Aux-событий 

        os.makedirs(self.temp_dir, exist_ok=True)

    def pipeline(self) -> Tuple:
        """
        загрузка датасета из файлов в каталоге self.temp_dir
        сбор DF
        получение дополнительных параметров:
        	типа скорости и ускорения изменения основных характеристик,
        	получение карт R-пиков и т.д.
        label-классификация, перевод в ohe
        нормирование/стандартизирование
        разделение X, y на выборки x_train, y_train, x_val, y_val, x_test, y_test        
        """
        
        self._check_channels_per_patient()                      # Формирует словарь доступных каналов пациенту
        self._analyze_channels_statistics()                     # Выбирает 2 самых популярных канала и выводит статистику по ним
        
        if self.check_analytics:
            # Анализ каналов ЭКГ
            self._visualize_channels_per_patient_table()            # Вывод таблицы: какие каналы есть у какого пациента
            self._visualize_channel_distribution()                  # График: распределение популярности каналов (barplot)
            self._visualize_patient_channel_matrix()                # Heatmap: матрица наличия каналов по пациентам

            # Добавление в аннотации ритмов из Aux
        self._add_rhythm_annotations()                          # Добавляет колонку Current_Rhythm в df_all_annotations на основе колонки Aux
        if self.check_analytics:
            self._analyze_Current_Rhythm_statistics()               # Формирует общую статистику по Aux-событиям
            self._analyze_patient_rhythm_type_stats()               # Формирует статистику Aux-событий по каждоиу пациенту
            self._binary_rhythm_type_analysis()                     # Анализирует баланс нормальных R-пиков в нормальных Aux-событиях
            self._visualize_global_rhythm_distribution()            # 
            self._visualize_rhythm_abnormal_distribution()          # 
            self._visualize_binary_rhythm_analysis()                # 

            # Анализ R-пиков и типов событий
            self._analyze_r_peak_statistics()                       # Формирует статистику по R-пикам и типам событий 
            self._visualize_global_peak_distribution()              # Pie / barplot: общее распределение типов R-пиков
            self._visualize_abnormal_peak_ratio()                   # Barplot: процент аномальных пиков на пациента
            self._visualize_patient_peak_types_heatmap()            # Heatmap: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='full')    # Color-bars: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='reduced') # Color-bars: аггрегированные топ-типы пиков у каждого пациента
            self._visualize_top_anomalies_pie()                     # Pie chart: самые частые аномалии по пациентам
        
        # Формирование окончательных датасетов
        self._filter_dataframes()                               # Формирование итоговых датафреймов под 2 задачи

        return (
            self.target_channel_name_1,
            self.target_channel_name_2,
            self.df_top_signals, 
            self.df_top_annotations, 
            self.df_cross_signals, 
            self.df_cross_annotations, 
            self.df_united_signals_1, 
            self.df_united_annotation_1, 
            self.df_united_signals_2, 
            self.df_united_annotation_2
        )
    
    def _check_channels_per_patient(self):
        """
        Проверяет, какие каналы доступны у каждого пациента.
        Формирует self.channels_per_patient — словарь:
            {pid: {'channels': [список], 'ohe': [вектор]}}
        """
        all_channels = ['MLII', 'V1', 'V2', 'V4', 'V5']

        for pid in tqdm(self.patient_ids, desc="Проверяем каналы", unit=" пациент"):
            df_p = self.df_all_signals[self.df_all_signals['Patient_id'] == pid]
            available_channels = [
                ch for ch in all_channels if ch in df_p.columns and not df_p[ch].isna().all()
            ]
            ohe_vector = [1 if ch in available_channels else 0 for ch in all_channels]
            self.channels_per_patient[pid] = {
                'channels': available_channels,
                'ohe': ohe_vector
            }

    def _analyze_channels_statistics(self):
        """
        Вспомогательный метод. Нужен для анализа данных и принятия решения о выборе каналов,
        на основе которых будет формироваться однородные данные будущего ДС.
        """

        channel_counter = Counter()
        for data in self.channels_per_patient.values():
            channel_counter.update(data['channels'])
        # === Таблица распределения каналов по пациентам ===
        print("\n=== Таблица распределения каналов по пациентам ===")
        print("Канал | Число пациентов с этим каналом")
        print("-" * 35)

        for ch, count in sorted(channel_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{ch.ljust(6)}| {count}")
        
        # === Автоматический выбор двух самых популярных каналов ===
        total_patients = int(len(self.patient_ids))
        # Получаем список каналов, отсортированных по числу пациентов (в порядке убывания)
        sorted_channels = sorted(channel_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Берём 1-й по популярности канал
        self.target_channel_name_1 = sorted_channels[0][0]
        channel_1_count = sorted_channels[0][1]
        channel_1_percentage = (channel_1_count / total_patients) * 100

        # Берём 2-й по популярности канал
        self.target_channel_name_2 = sorted_channels[1][0]
        channel_2_count = sorted_channels[1][1]
        channel_2_percentage = (channel_2_count / total_patients) * 100

        # === Вывод результатов ===
        print("\n=== Лидирующие каналы ===")
        print(f"Наиболее распространённый канал: {self.target_channel_name_1}")
        print(f"Число пациентов: {channel_1_count} из {total_patients}")
        print(f"Процент от всех пациентов: {channel_1_percentage:.2f}%")

        print(f"Второй по распространённости канал: {self.target_channel_name_2}")
        print(f"Число пациентов: {channel_2_count} из {total_patients}")
        print(f"Процент от всех пациентов: {channel_2_percentage:.2f}%")

        print(f"\nВЫВОД:\nОчевидно, что канал: {self.target_channel_name_1} будет выбран в качестве источника сигналов, формирующих датасет.\nЕго показания есть у {channel_1_percentage:.2f}% пациентов.\nДанными оставшихся {total_patients - channel_1_count} ({total_patients} - {channel_1_count}) пациентов можно пожертвовать ради однородности и чистоты датасета.")
        print(f"\nКанал {self.target_channel_name_2} можно выбирать для сравнения\nЕго показания есть у {channel_2_percentage:.2f}% пациентов.\nПри сравнении придется пожертвовать данными уже {total_patients - channel_2_count} ({total_patients} - {channel_2_count}) пациентов.")

    def _visualize_channels_per_patient_table(self):
        """
        Выводит DataFrame: пациент | количество каналов | список каналов
        """
        data = []
        for pid, info in self.channels_per_patient.items():
            data.append({
                'Patient_id': pid,
                'Channel_count': len(info['channels']),
                'Channels': ', '.join(info['channels'])
            })

        df_summary = pd.DataFrame(data)
        print("\n=== Таблица каналов по пациентам ===")
        print(df_summary.to_string(index=False))

    def _visualize_channel_distribution(self):
        """
        График: сколько пациентов имеют каждый из каналов
        """
        channel_counter = {}
        for info in self.channels_per_patient.values():
            for ch in info['channels']:
                channel_counter[ch] = channel_counter.get(ch, 0) + 1

        # Сортируем по убыванию
        channel_counter = dict(sorted(channel_counter.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(channel_counter.keys()), y=list(channel_counter.values()))
        plt.title('Распределение каналов по числу пациентов')
        plt.ylabel('Число пациентов')
        plt.xlabel('Каналы ЭКГ')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "channel_distribution.png"))
        plt.show()

    def _visualize_patient_channel_matrix(self):
        """
        Heatmap: матрица пациентов и каналов (1 - есть, 0 - нет)
        """
        all_channels = ['MLII', 'V1', 'V2', 'V4', 'V5']
        data = []

        for pid, info in self.channels_per_patient.items():
            row = {ch: 1 if ch in info['channels'] else 0 for ch in all_channels}
            row['Patient_id'] = pid
            data.append(row)

        df_matrix = pd.DataFrame(data).set_index('Patient_id')

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_matrix, annot=True, cmap='Blues', cbar=False)
        plt.title('Наличие каналов по пациентам')
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "patient_channel_matrix.png"))
        plt.show()

    def _add_rhythm_annotations(self):
        """
        Добавляет колонку 'Current_Rhythm' в self.df_all_annotations.
        Значение распространяется до следующего изменения ритма.
        """

        # Извлекаем только строки с ритмическими аннотациями
        rhythm_mask = self.df_all_annotations['Aux'].notna() & self.df_all_annotations['Aux'].str.startswith('(')
        rhythm_df = self.df_all_annotations.loc[rhythm_mask, ['Sample', 'Patient_id', 'Aux']].copy()

        # Убираем скобку '(' из значения
        rhythm_df['Current_Rhythm'] = rhythm_df['Aux'].str[1:]  # например, '(AFIB' → 'AFIB'

        # Удаляем исходную колонку Aux (если не нужна далее)
        rhythm_df.drop(columns=['Aux'], inplace=True)

        # Объединяем обратно по Patient_id и Sample
        # НО нам нужно заполнить NaN вниз — forward fill по каждому пациенту
        self.df_all_annotations = pd.merge(
            self.df_all_annotations,
            rhythm_df[['Sample', 'Patient_id', 'Current_Rhythm']],
            on=['Sample', 'Patient_id'],
            how='left'
        )

        # Заполняем пропуски в Current_Rhythm
        # Сортируем по Patient_id и Sample, чтобы заполнение шло по времени
        self.df_all_annotations.sort_values(['Patient_id', 'Sample'], inplace=True)
        self.df_all_annotations['Current_Rhythm'] = (
            self.df_all_annotations.groupby('Patient_id', group_keys=False)['Current_Rhythm'].apply(lambda x: x.ffill())
        )

        #print("Колонка 'Current_Rhythm' добавлена к аннотациям.")

    def _analyze_Current_Rhythm_statistics(self):
        """
        Анализирует распределение событий в поле 'Current_Rhythm'
        Формирует статистику: частота встречаемости Current_Rhythm-меток
        """

        # --- Убедимся, что 'Current_Rhythm' присутствует ---
        if 'Current_Rhythm' not in self.df_all_annotations.columns:
            print("Колонка 'Current_Rhythm' отсутствует в аннотациях")
            return

        # Берём только колонку 'Current_Rhythm'
        rhythm_series = self.df_all_annotations['Current_Rhythm']

        # Убираем NaN
        rhythm_series = rhythm_series[rhythm_series.notna()]

        # Считаем встречаемость каждого ритма
        rhythm_counts = rhythm_series.value_counts()
        total = len(rhythm_series)
        rhythm_percent = (rhythm_counts / total * 100).round(2)

        # Сохраняем в датафрейм
        self.df_global_aux_distribution = pd.DataFrame({
            'Count': rhythm_counts,
            'Percent': rhythm_percent
        })

        # Сохраняем словарь типов Aux-событий
        self.rhythm_types = rhythm_series.unique().tolist()

        print("\n=== Распределение ритмов из Current_Rhythm ===")
        print(self.df_global_aux_distribution.sort_values(by='Count', ascending=False))


    def _analyze_patient_rhythm_type_stats(self):
        """
        Собирает статистику распределения Type внутри Current_Rhythm на пациента.
        Формирует датафрейм self.df_rhythm_type_distribution
        """

        all_stats = []

        for pid in tqdm(self.patient_ids, desc="Распределение Type в Current_Rhythm", unit=" пациент"):
            df_p = self.df_all_annotations[self.df_all_annotations['Patient_id'] == pid]

            for rhythm in self.rhythm_types:
                df_r = df_p[df_p['Current_Rhythm'] == rhythm]

                if df_r.empty:
                    continue

                total_peaks = len(df_r)
                normal_count = len(df_r[df_r['Type'] == 'N'])
                abnormal_count = len(df_r[df_r['Type'] != 'N'])

                stats = {
                    'Patient_id': pid,
                    'Rhythm': rhythm,
                    'Total_peaks': total_peaks,
                    'Normal_count': normal_count,
                    'Normal_percent': round((normal_count / total_peaks * 100), 2),
                    'Abnormal_count': abnormal_count,
                    'Abnormal_percent': round((abnormal_count / total_peaks * 100), 2),
                }

                # Можно добавить и конкретные типы аномалий
                anomaly_types = ['V', 'A', 'F', 'Q', 'L', 'R']
                for atype in anomaly_types:
                    count = len(df_r[df_r['Type'] == atype])
                    stats[f'{atype}_count'] = count
                    stats[f'{atype}_percent'] = round((count / total_peaks * 100), 2)

                all_stats.append(stats)

        # Сохраняем как датафрейм
        self.df_rhythm_type_distribution = pd.DataFrame(all_stats).fillna(0).astype({
            'Patient_id': str,
            'Rhythm': str,
            'Total_peaks': int,
            **{f'{t}_count': int for t in ['Normal', 'Abnormal', 'V', 'A', 'F', 'Q', 'L', 'R']}
        })

        print("\n=== Распределение типов пиков внутри ритмов ===")
        print(self.df_rhythm_type_distribution.head())

    def _binary_rhythm_type_analysis(self):
        """
        Проводит бинарный анализ:
            - Категория 0: Type == 'N' и Current_Rhythm == 'N' → "чистая норма"
            - Категория 1: все остальные случаи → "аномалия"
        
        Выводит статистику по всему датасету.
        """

        if 'Current_Rhythm' not in self.df_all_annotations.columns:
            print("Колонка 'Current_Rhythm' отсутствует — невозможно провести бинарный анализ")
            return

        # --- Подсчёт категорий ---
        mask_normal = (self.df_all_annotations['Type'] == 'N') & \
                    (self.df_all_annotations['Current_Rhythm'] == 'N')

        normal_count = mask_normal.sum()
        abnormal_count = len(self.df_all_annotations) - normal_count

        total = normal_count + abnormal_count

        percent_normal = round((normal_count / total) * 100, 2)
        percent_abnormal = round((abnormal_count / total) * 100, 2)

        # --- Вывод ---
        print("\n=== Бинарная статистика ===")
        print(f"Чисто N в ритме N: {normal_count} ({percent_normal}%)")
        print(f"Всё остальное (в т.ч. N в других ритмах): {abnormal_count} ({percent_abnormal}%)\n")

        print("Пример распределения:")
        print(self.df_all_annotations.loc[mask_normal.head().index, ['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])
        print("...")
        print(self.df_all_annotations.loc[~mask_normal].head()[['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])

    def _visualize_global_rhythm_distribution(self):
        """
        График: общее распределение ритмов из Current_Rhythm (в процентах)
        """

        df = self.df_global_aux_distribution.sort_values(by='Count', ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=df.index, y='Percent', data=df, palette='viridis')
        plt.title("Распределение ритмов из Current_Rhythm (в % от всех пиков)")
        plt.ylabel("Процент")
        plt.xlabel("Ритм")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "global_rhythm_distribution.png"))
        plt.show()

    def _visualize_rhythm_abnormal_distribution(self):
        """
        График: доля аномальных пиков внутри каждого ритма
        """

        # Фильтрация по ритму и группировка
        rhythm_summary = (
            self.df_rhythm_type_distribution.groupby('Rhythm')[['Normal_percent', 'Abnormal_percent']]
            .mean()
            .sort_values(by='Abnormal_percent', ascending=False)
        )

        rhythm_summary.plot(kind='barh', stacked=True, figsize=(12, 8), color=['#4CAF50', '#F44336'])
        plt.title("Доля нормальных и аномальных пиков внутри ритмов (в %)")
        plt.xlabel("Процент от всех пиков в ритме")
        plt.ylabel("Ритм")
        plt.legend(['Нормальные', 'Аномальные'])
        plt.gca().invert_yaxis()  # чтобы самые аномальные были сверху
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "rhythm_abnormal_distribution.png"))
        plt.show()

    def _visualize_binary_rhythm_analysis(self):
        """
        Pie chart: соотношение 'чистых N в N' к 'всему остальному'
        """

        normal_count = len(self.df_all_annotations[
            (self.df_all_annotations['Type'] == 'N') &
            (self.df_all_annotations['Current_Rhythm'] == 'N')
        ])

        abnormal_count = len(self.df_all_annotations) - normal_count

        labels = ['Чистая норма (N+N)', 'Всё остальное']
        sizes = [normal_count, abnormal_count]
        colors = ['#4CAF50', '#F44336']

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title("Баланс 'чистой нормы' и 'всего прочего'")
        plt.axis('equal')  # круглый пирог
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "binary_rhythm_analysis_pie.png"))
        plt.show()

    def _analyze_r_peak_statistics(self):
        """
        Проводит анализ R-пиков и формирует датафреймы только с относительными характеристиками:
            - global_peak_distribution: общий профиль (оставлены Count + Percent)
            - patient_peak_top_types: топ-5 типов пиков у каждого пациента (только проценты)
            - patient_normal_abnormal_ratio: доля нормальных / аномальных пиков (только проценты)
            - patient_top_anomaly: самый частый аномальный пик (только процент от аномалий)
        """

        # === 1. Общее распределение типов R-пиков (сохраняем Count + Percent) ===
        total_counts = self.df_all_annotations['Type'].value_counts()
        total_percent = (total_counts / total_counts.sum()) * 100

        self.df_global_peak_distribution = pd.DataFrame({
            'Count': total_counts,
            'Percent': total_percent.round(2)
        })

        # === 2. Топ-5 типов пиков по каждому пациенту (только проценты) ===
        patient_peak_top_types = []

        for pid in self.df_all_annotations['Patient_id'].unique():
            df_p = self.df_all_annotations[self.df_all_annotations['Patient_id'] == pid]
            type_counts = df_p['Type'].value_counts()
            total = len(df_p)

            row = {'Patient_id': pid}
            for i, (peak_type, count) in enumerate(type_counts.head(5).items()):
                percent = round((count / total) * 100, 2)
                row[f'Type_{i+1}'] = peak_type
                row[f'Percent_{i+1}'] = percent

            patient_peak_top_types.append(row)

        self.df_patient_peak_top_types = pd.DataFrame(patient_peak_top_types)

        # === 3. Нормальные vs Аномальные пики по пациентам (только проценты) ===
        #anomaly_classes = ['V', 'A', 'F', 'J', 'S', '/', 'E', 'Q']

        normal_abnormal_data = []
        for pid in self.df_all_annotations['Patient_id'].unique():
            df_p = self.df_all_annotations[self.df_all_annotations['Patient_id'] == pid]
            total = len(df_p)

            normal_count = df_p[df_p['Type'] == 'N'].shape[0]
            abnormal_count = total - normal_count

            normal_abnormal_data.append({
                'Patient_id': pid,
                'Normal': round((normal_count / total) * 100, 2),
                'Abnormal': round((abnormal_count / total) * 100, 2)
            })

        self.df_patient_normal_abnormal_ratio = pd.DataFrame(normal_abnormal_data)

        # === 4. Самый частый аномальный пик у пациента (только процент от всех аномалий) ===
        top_anomalies = []
        for pid in self.df_all_annotations['Patient_id'].unique():
            df_p = self.df_all_annotations[self.df_all_annotations['Patient_id'] == pid]
            anomalies = df_p[df_p['Type'] != 'N']['Type'].value_counts()

            if not anomalies.empty:
                top_anomaly = anomalies.index[0]
                percent = round((anomalies.iloc[0] / anomalies.sum()) * 100, 2)
            else:
                top_anomaly = '-'
                percent = 0

            top_anomalies.append({
                'Patient_id': pid,
                'Top_anomaly': top_anomaly,
                'Top_anomaly_percent': percent
            })

        self.df_patient_top_anomaly = pd.DataFrame(top_anomalies)

    def _visualize_global_peak_distribution(self):
        df = self.df_global_peak_distribution.sort_values(by='Count', ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        plt.bar(df.index, df['Percent'], color=[self._get_color_for_peak_type(t) for t in df.index])
        plt.title("Распределение типов R-пиков (в % от общего числа)")
        plt.ylabel("Процент")
        plt.xlabel("Тип пика")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "global_peak_distribution.png"))
        plt.show()

    def _visualize_abnormal_peak_ratio(self):
        df = self.df_patient_normal_abnormal_ratio.sort_values(by='Abnormal', ascending=False)
        
        plt.figure(figsize=(14, 8))
        plt.bar(df['Patient_id'], df['Abnormal'], color='steelblue')
        plt.title("Доля аномальных R-пиков по пациентам")
        plt.ylabel("Процент аномальных пиков")
        plt.xlabel("Пациент ID")
        plt.xticks(rotation=90)
        plt.axhline(df['Abnormal'].mean(), color='black', linestyle='--', label=f'Среднее: {df["Abnormal"].mean():.1f}%')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "abnormal_peak_ratio.png"))
        plt.show()

    def _visualize_patient_peak_types_heatmap(self):
        """
        Heatmap: топ-типы пиков у каждого пациента
        """

        # Формируем матрицу: пациент × топ-5 типов
        df = self.df_patient_peak_top_types.set_index('Patient_id')

        # Берём только типы пиков
        peak_types = df.filter(like='Type_', axis=1)

        # Создаём бинарную матрицу: 1 — тип присутствует у пациента, 0 — нет
        binary_matrix = peak_types.fillna('-').apply(lambda x: x != '-', 1).astype(int)

        plt.figure(figsize=(10, 8))
        sns.heatmap(binary_matrix, cmap='Blues', cbar=False, annot=peak_types.fillna(''), fmt='', linewidths=.5)
        plt.title("Топ-5 типов R-пиков по пациентам (Heatmap)")
        plt.ylabel("Пациент")
        plt.xlabel("Позиция в рейтинге типа")
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "patient_peak_types_heatmap.png"))
        plt.show()

    def _get_color_for_peak_type(self, peak_type, mode='full'):
        """
        Возвращает цвет для типа события.
        Если mode='reduced' → возвращает цвет по группе: Normal / Abnormal / BBB / Noise
        """

        full_color_map = {
            'N': '#4CAF50',     # зелёный
            'V': '#F44336',     # красный
            'A': '#2196F3',     # синий
            'F': '#FF9800',     # оранжевый
            'J': '#9C27B0',     # фиолетовый
            'S': '#FFEB3B',     # жёлтый
            '/': '#E91E63',     # розовый
            'Q': '#9E9E9E',     # серый
            'L': '#03A9F4',     # голубой
            'R': '#00BCD4',     # циан
            '+': '#FF5722',     # deep orange
            '~': '#795548',     # brown
            '[': '#795548',     # brown
            '!': '#795548',     # brown
            ']': '#795548',
            '"': '#795548',
            '?': '#795548',
            'x': '#795548',
            'f': '#795548',
            'other': '#EEEEEE'
        }

        reduced_color_map = {
            'Normal': '#4CAF50',  # зелёный
            'Abnormal': '#F44336',  # красный
            'BBB': '#03A9F4',   # голубой
            'Noise': '#FF5722'   # deep orange
        }

        if mode == 'full':
            return full_color_map.get(peak_type, '#DDDDDD')
        elif mode == 'reduced':
            # Сопоставляем конкретные типы с категориями
            return reduced_color_map[peak_type]
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

    def _visualize_patient_peak_types_bars(self, mode='full'):
        """
        Визуализирует топ-типы R-пиков по пациентам в виде горизонтальных столбцов.
        
        :param mode: 
            'full' → все типы событий (исходные)
            'reduced' → три категории: N, Abnormal, Noise
        """

        plt.figure(figsize=(14, len(self.df_patient_peak_top_types) * 0.5))  # адаптируем высоту под число пациентов

        # --- Сортируем пациентов ---
        df = self.df_patient_peak_top_types.sort_values(by='Patient_id')

        pids = df['Patient_id'].tolist()

        # --- Категории для reduced mode ---
        normal_types = ['N']
        abnormal_types = ['V', 'A', 'F', 'J', 'S', 'E', 'Q']
        blocade_types = ['L', 'R']
        noise_types = ['+', '/', '~', '[', '!', ']', '"', '?', 'x', 'f']

        for i, row in enumerate(df.to_dict(orient='records')):
            types = []
            percents = []

            if mode == 'full':
                for j in range(1, 6):
                    peak_type = row[f'Type_{j}']
                    percent = row[f'Percent_{j}']
                    if pd.notna(peak_type) and percent > 0:
                        types.append(peak_type)
                        percents.append(percent)

            elif mode == 'reduced':
                merged = {
                    'Normal': 0,
                    'Abnormal': 0,
                    'BBB': 0,
                    'Noise': 0
                }

                for j in range(1, 6):
                    peak_type = row[f'Type_{j}']
                    percent = row[f'Percent_{j}']
                    if pd.isna(peak_type) or percent <= 0:
                        continue

                    if peak_type in normal_types:
                        merged['Normal'] += percent
                    elif peak_type in abnormal_types:
                        merged['Abnormal'] += percent
                    elif peak_type in blocade_types:
                        merged['BBB'] += percent
                    elif peak_type in noise_types:
                        merged['Noise'] += percent

                for k, v in merged.items():
                    if v > 0:
                        types.append(k)
                        percents.append(v)

            # --- Получаем цвета ---
            colors = [self._get_color_for_peak_type(t, mode=mode) for t in types]

            # --- Рисуем bar ---
            left = 0
            for p, color in zip(percents, colors):
                plt.barh(i, p, left=left, color=color, edgecolor='black', height=0.7)
                left += p

            # --- Подписываем ID пациента слева ---
            plt.text(-5, i, str(row['Patient_id']), va='center', ha='right', fontsize=10)

        # --- Настройка графика ---
        plt.yticks(np.arange(len(pids)), [])
        plt.xlabel("Процент от всех пиков")
        plt.title("Распределение типов R-пиков по пациентам (в %)")
        plt.xlim(0, 100)
        plt.tight_layout()

        # --- Легенда зависит от режима ---
        legend_patches = []

        if mode == 'full':
            used_types = df.filter(like='Type_', axis=1).apply(pd.Series.unique).values[0]
            for t in sorted(set(used_types)):
                if pd.isna(t) or t == '-':
                    continue
                color = self._get_color_for_peak_type(t, mode='full')
                legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=color, label=t))

        elif mode == 'reduced':
            for t in ['Normal', 'Abnormal', 'BBB', 'Noise']:
                color = self._get_color_for_peak_type(t, mode='reduced')
                legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=color, label=t))
            
        plt.legend(
            handles=legend_patches,
            title="Типы событий",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10
        )

        plt.savefig(os.path.join(self.temp_dir, f"patient_peak_types_bars_{mode}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def _visualize_top_anomalies_pie(self):
        """
        Pie chart: самые частые аномалии у пациентов
        """

        df = self.df_patient_top_anomaly

        # Считаем, сколько раз какой тип встречается как топ-аномалия
        anomaly_counter = df['Top_anomaly'].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(anomaly_counter, labels=anomaly_counter.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
        plt.title("Наиболее частые аномалии у пациентов")
        plt.axis('equal')  # круглый пирог
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "top_anomalies_pie.png"))
        plt.show()

    def _filter_dataframes(self):
        """
        Формирует датафреймы сигналов и аннотаций по двум каналам:
            - self.df_top_signals / self.df_top_annotations → обучение на основном канале (target_channel_name_1)
            - self.df_cross_signals / self.df_cross_annotations → тестирование на втором канале (target_channel_name_2)
            - self.df_united_signals_1 / self.df_united_annotation_1 → данные для модели 1
            - self.df_united_signals_2 / self.df_united_annotation_2 → данные для модели 2
        """

        # Фильтруем по первому каналу
        self.df_top_signals, self.df_top_annotations = self._filter_df(
            self.df_all_signals,
            self.df_all_annotations,
            self.target_channel_name_1
        )
        print(f"Оставлено записей после фильтрации (по {self.target_channel_name_1}): {len(self.df_top_signals)}")
        print(f"Оставлено аннотаций после фильтрации (по {self.target_channel_name_1}): {len(self.df_top_annotations)}")

        # Фильтруем по второму каналу
        self.df_cross_signals, self.df_cross_annotations = self._filter_df(
            self.df_all_signals,
            self.df_all_annotations,
            self.target_channel_name_2
        )
        print(f"Оставлено записей после фильтрации (по {self.target_channel_name_2}): {len(self.df_cross_signals)}")
        print(f"Оставлено аннотаций после фильтрации (по {self.target_channel_name_2}): {len(self.df_cross_annotations)}")

        # Находим общих пациентов
        common_pids = np.intersect1d(
            self.df_top_signals['Patient_id'].unique(),
            self.df_cross_signals['Patient_id'].unique()
        )

        # Двойная фильтрация по первому каналу
        self.df_united_signals_1 = self.df_top_signals[self.df_top_signals['Patient_id'].isin(common_pids)]
        self.df_united_annotation_1 = self.df_top_annotations[self.df_top_annotations['Patient_id'].isin(common_pids)]

        # Двойная фильтрация по второму каналу
        self.df_united_signals_2 =  self.df_cross_signals[self.df_cross_signals['Patient_id'].isin(common_pids)]
        self.df_united_annotation_2 =  self.df_cross_annotations[self.df_cross_annotations['Patient_id'].isin(common_pids)]
        print(f"Оставлено записей после 2-ной фильтрации (по {self.target_channel_name_1} и {self.target_channel_name_2}): {len(self.df_united_signals_1)}")
        print(f"Оставлено аннотаций после 2-ной фильтрации (по {self.target_channel_name_1} и {self.target_channel_name_2}): {len(self.df_united_annotation_1)}")

    def _filter_df(self, df_s, df_a, channel):
        """
        Фильтрует сигналы и аннотации по наличию данных в указанном канале.

        :param df_s: исходный датафрейм сигналов (pd.DataFrame)
        :param df_a: исходный датафрейм аннотаций (pd.DataFrame)
        :param channel: название канала для фильтрации (например 'MLII', 'V1')
        :return: df_filtered_s, df_filtered_a
        """

        # Фильтруем сигналы по наличию данных в целевом канале
        df_filtered_s = (df_s[df_s[channel].notna()][['Sample', channel, 'Patient_id']])

        # Получаем уникальные pid из отфильтрованных сигналов
        valid_pids = df_filtered_s['Patient_id'].unique()

        # Фильтруем аннотации
        df_filtered_a = df_a[df_a['Patient_id'].isin(valid_pids)]

        return df_filtered_s, df_filtered_a