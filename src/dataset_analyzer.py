# src/dataset_analyzer.py

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from tabulate import tabulate

from tqdm import tqdm
from .config import config
from typing import TYPE_CHECKING

# Для обещания типизации, чтобы не создавать циклический вызов классов: DatasetPreprocessing <-> DatasetAnalyze
if TYPE_CHECKING:
    from .dataset_preprocessing import DatasetPreprocessing

class DatasetAnalyze:
    """
    Отвечает за исследовательский анализ данных (EDA) и генерацию визуализаций.
    
    Принимает контейнер с данными от DatasetPreprocessing и выполняет следующие задачи:
    - Анализирует и визуализирует распределение ЭКГ-каналов по пациентам.
    - Анализирует и визуализирует распределение типов ритмов и R-пиков.
    - Генерирует и сохраняет детализированную сводку по пациентам в JSON для GUI.
    """
    def __init__(self, data_container: 'DatasetPreprocessing') -> None:
        """
        Инициализирует анализатор.
        
        :param data_container: Экземпляр класса DatasetPreprocessing, содержащий все загруженные данные.
        """
        self.container = data_container                         # self от внешнего класса DatasetPreprocessing
        self.temp_dir = config['paths']['temp_dir']
        self.check_analytics = config['data']['analytics']

        # Результаты работы АНАЛИЗАТОРА (внутренний self)
        self.df_global_peak_distribution = pd.DataFrame()       # Общее распределение типов R-пиков: Count, Percent
        self.df_patient_peak_top_types = pd.DataFrame()         # Топ-5 типов пиков на пациента → только проценты
        self.df_patient_normal_abnormal_ratio = pd.DataFrame()  # Процент нормальных / аномальных пиков на пациента
        self.df_patient_top_anomaly = pd.DataFrame()            # Самый частый аномальный пик у каждого пациента + его доля
        self.rhythm_types = {}                                  # Словарь Aux-событий 
        self.peak_statistics_for_stage2 = pd.DataFrame()        # Распределение R-пиков на 2й стадии (без N+N)
        self.df2_all_annotations = None                         # Все аннотации кроме N: по всем каналам  (для анализа ДС)                              

        os.makedirs(self.temp_dir, exist_ok=True)

    def run(self) -> None:
        """
        Запускает полный цикл исследовательского анализа данных (EDA).
        Строит графики и выводит в консоль таблицы со статистикой
        по распределению каналов, ритмов и типов R-пиков.
        Работает с данными, уже загруженными и отфильтрованными на предыдущих этапах.
        """

        if self.check_analytics:
            # Анализ каналов ЭКГ
            self._visualize_channels_per_patient_table()                        # Вывод таблицы: какие каналы есть у какого пациента
            self._visualize_channel_distribution()                              # График: распределение популярности каналов (barplot)
            self._visualize_patient_channel_matrix()                            # Heatmap: матрица наличия каналов по пациентам

            self._analyze_Current_Rhythm_statistics(self.container.df_all_annotations)    # Формирует общую статистику по Aux-событиям
            self._analyze_patient_rhythm_type_stats()                           # Формирует статистику Aux-событий по каждоиу пациенту
            self._binary_rhythm_type_analysis()                                 # Анализирует баланс нормальных R-пиков в нормальных Aux-событиях
            self._visualize_global_rhythm_distribution()                        # График: распределение ритмов (barplot)
            self._visualize_rhythm_abnormal_distribution()                      # График: доля аномальных пиков внутри ритмов
            self._visualize_binary_rhythm_analysis()                            # Pie chart: баланс "чистой нормы" к остальным

            # Анализ R-пиков и типов событий
            self._analyze_r_peak_statistics()                                   # Формирует статистику по R-пикам и типам событий 
            self._visualize_global_peak_distribution()                          # Pie / barplot: общее распределение типов R-пиков
            self._visualize_abnormal_peak_ratio()                               # Barplot: процент аномальных пиков на пациента
            self._visualize_patient_peak_types_heatmap()                        # Heatmap: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='full')                # Color-bars: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='reduced')             # Color-bars: аггрегированные топ-типы пиков у каждого пациента
            self._visualize_top_anomalies_pie()                                 # Pie chart: самые частые аномалии по пациентам
        
        self._analyze_class_distribution_by_patient()                           # Определение данных для стратификации редких R-пиков по пациентам между выборками T/V/T.
        ### self.analyze_split_balance()                                            # Исследование распределения классов после ручной стратификации
        ### self.create_patient_profiles_table()                                    

        # 2я стадия выделение датафреймов без N+N R-пиков
        self._create_dataframes_for_stage_2()                                   # создает self.df2_all_signals и self.df2_all_annotations

        if self.check_analytics:
            # Проверяем, что датафрейм для stage 2 был успешно создан
            if self.df2_all_annotations is not None:
                # Анализ R-пиков и типов событий
                self._analyze_Current_Rhythm_statistics(self.df2_all_annotations)   # Формирует общую статистику по Aux-событиям
                self._binary_rhythm_type_analysis_for_stage2(self.df2_all_annotations)  # Анализирует баланс нормальных R-пиков вне нормальных Aux-событиий (стадия 2)
                self._visualize_global_rhythm_distribution(stage='stage_2_')        # График: распределение ритмов (barplot)
                self._visualize_binary_rhythm_analysis_for_stage2()                 # Pie chart: баланс оставшихся N к не-N
                self._analyze_peak_statistics_for_stage2()                          # Формирует общую статистику распределению R-пиков на 2й стадии (без N+N)
                self._visualize_all_peak_types_for_stage2()                         # Визуализирует общую статистику распределению R-пиков на 2й стадии (без N+N)
            else:
                print("ПРЕДУПРЕЖДЕНИЕ: DataFrame df2_all_annotations не был создан, анализ для Stage 2 пропускается.")

    def _visualize_channels_per_patient_table(self) -> None:
        """
        Выводит DataFrame: пациент | количество каналов | список каналов
        """
        data = []
        for pid, info in self.container.channels_per_patient.items():
            data.append({
                'Patient_id': pid,
                'Channel_count': len(info['channels']),
                'Channels': ', '.join(info['channels'])
            })

        df_summary = pd.DataFrame(data)
        print("\n=== Таблица каналов по пациентам ===")
        print(df_summary.to_string(index=False))

    def _visualize_channel_distribution(self) -> None:
        """
        График: сколько пациентов имеют каждый из каналов
        """
        channel_counter = {}
        for info in self.container.channels_per_patient.values():
            for ch in info['channels']:
                channel_counter[ch] = channel_counter.get(ch, 0) + 1

        # Сортируем по убыванию
        channel_counter = dict(sorted(channel_counter.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(channel_counter.keys()), y=list(channel_counter.values()), hue=list(channel_counter.keys()))
        plt.title('Распределение каналов по числу пациентов')
        plt.ylabel('Число пациентов')
        plt.xlabel('Каналы ЭКГ')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "channel_distribution.png"))
        plt.show()

    def _visualize_patient_channel_matrix(self) -> None:
        """
        Heatmap: матрица пациентов и каналов (1 - есть, 0 - нет)
        """
        all_channels = ['MLII', 'V1', 'V2', 'V4', 'V5']
        data = []

        for pid, info in self.container.channels_per_patient.items():
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

    def _analyze_Current_Rhythm_statistics(self, df_all_annotations: pd.DataFrame) -> None:
        """
        Анализирует распределение событий в поле 'Current_Rhythm'
        Формирует статистику: частота встречаемости Current_Rhythm-меток
        """

        # --- Убедимся, что 'Current_Rhythm' присутствует ---
        if 'Current_Rhythm' not in df_all_annotations.columns:
            print("Колонка 'Current_Rhythm' отсутствует в аннотациях")
            return

        # Берём только колонку 'Current_Rhythm'
        rhythm_series = df_all_annotations['Current_Rhythm']

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

    def _analyze_patient_rhythm_type_stats(self) -> None:
        """
        Собирает статистику распределения Type внутри Current_Rhythm на пациента.
        Формирует датафрейм self.df_rhythm_type_distribution
        """

        all_stats = []

        for pid in tqdm(self.container.patient_ids, desc="Распределение Type в Current_Rhythm", unit=" пациент"):
            df_p = self.container.df_all_annotations[self.container.df_all_annotations['Patient_id'] == pid]

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

    def _binary_rhythm_type_analysis(self) -> None:
        """
        Проводит бинарный анализ:
            - Категория 0: Type == 'N' и Current_Rhythm == 'N' → "чистая норма"
            - Категория 1: все остальные случаи → "аномалия"
        
        Выводит статистику по всему датасету.
        """

        if 'Current_Rhythm' not in self.container.df_all_annotations.columns:
            print("Колонка 'Current_Rhythm' отсутствует — невозможно провести бинарный анализ")
            return

        # --- Подсчёт категорий ---
        mask_normal = (self.container.df_all_annotations['Type'] == 'N') & \
                    (self.container.df_all_annotations['Current_Rhythm'] == 'N')

        normal_count = mask_normal.sum()
        abnormal_count = len(self.container.df_all_annotations) - normal_count

        total = normal_count + abnormal_count

        percent_normal = round((normal_count / total) * 100, 2)
        percent_abnormal = round((abnormal_count / total) * 100, 2)

        # --- Вывод ---
        print("\n=== Бинарная статистика ===")
        print(f"Чисто N в ритме N: {normal_count} ({percent_normal}%)")
        print(f"Всё остальное (в т.ч. N в других ритмах): {abnormal_count} ({percent_abnormal}%)\n")

        print("Пример распределения:")
        print(self.container.df_all_annotations.loc[mask_normal.head().index, ['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])
        print("...")
        print(self.container.df_all_annotations.loc[~mask_normal].head()[['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])

    def _binary_rhythm_type_analysis_for_stage2(self, df_all_annotations: pd.DataFrame) -> None:
        """
        Проводит бинарный анализ для данных стадии 2.
        Разделяет пики на три категории: 'N+N' (отфильтрованные), 'N' в аномальном ритме и 'не-N' пики.
        Выводит статистику по этим категориям.

        :param df_all_annotations: DataFrame с аннотациями для анализа (обычно self.df2_all_annotations).
        """
        if 'Current_Rhythm' not in df_all_annotations.columns:
            print("Колонка 'Current_Rhythm' отсутствует — невозможно провести бинарный анализ")
            return

        # --- Подсчёт категорий ---
        mask_normal_N = (df_all_annotations['Type'] == 'N') & (df_all_annotations['Current_Rhythm'] == 'N')
        mask_normal = (df_all_annotations['Type'] == 'N') & (df_all_annotations['Current_Rhythm'] != 'N')
        mask_not_normal = df_all_annotations['Type'] != 'N'

        normal_N_count = mask_normal_N.sum()
        normal_count = mask_normal.sum()
        abnormal_count = len(df_all_annotations) - normal_count - normal_N_count

        total = normal_count + abnormal_count + normal_N_count

        percent_normal_N = round((normal_N_count / total) * 100, 2)
        percent_normal = round((normal_count / total) * 100, 2)
        percent_abnormal = round((abnormal_count / total) * 100, 2)

        # --- Вывод ---
        print("\n=== Бинарная статистика ===")
        print(f"Чисто N+N: {normal_N_count} ({percent_normal_N}%)")
        print(f"Чисто N (при не-N Aux): {normal_count} ({percent_normal}%)")
        print(f"Всё остальное (не N): {abnormal_count} ({percent_abnormal}%)\n")

        print("Пример распределения:")
        print(df_all_annotations.loc[mask_normal_N.head().index, ['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])
        print("...")
        print(df_all_annotations.loc[mask_normal.head().index, ['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])
        print("...")
        print(df_all_annotations.loc[mask_not_normal].head()[['Patient_id', 'Sample', 'Type', 'Current_Rhythm']])

    def _visualize_global_rhythm_distribution(self, stage: str = '') -> None:
        """
        График: общее распределение ритмов из Current_Rhythm (в процентах)
        """

        df = self.df_global_aux_distribution.sort_values(by='Count', ascending=False).head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=df.index, y='Percent', data=df, palette='viridis', hue=df.index)
        plt.title("Распределение ритмов из Current_Rhythm (в % от всех пиков)")
        plt.ylabel("Процент")
        plt.xlabel("Ритм")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, f"{stage}global_rhythm_distribution.png"))
        plt.show()

    def _visualize_rhythm_abnormal_distribution(self) -> None:
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

    def _visualize_binary_rhythm_analysis(self) -> None:
        """
        Pie chart: соотношение 'чистых N в N' к 'всему остальному'
        """
        # --- ОТЛАДОЧНЫЙ БЛОК
        print("\n--- DEBUG: _visualize_binary_rhythm_analysis ---")
        print(f"Размер df_all_annotations: {self.container.df_all_annotations.shape}")
        print(f"Количество уникальных пациентов: {self.container.df_all_annotations['Patient_id'].nunique()}")

        normal_count = len(self.container.df_all_annotations[
            (self.container.df_all_annotations['Type'] == 'N') &
            (self.container.df_all_annotations['Current_Rhythm'] == 'N')
        ])

        abnormal_count = len(self.container.df_all_annotations) - normal_count

        total_for_pie = normal_count + abnormal_count
        percent_normal = (normal_count / total_for_pie) * 100 if total_for_pie > 0 else 0
        percent_abnormal = (abnormal_count / total_for_pie) * 100 if total_for_pie > 0 else 0

        print(f"Количество 'Чистая норма (N+N)': {normal_count} ({percent_normal:.2f}%)")
        print(f"Количество 'Всё остальное': {abnormal_count} ({percent_abnormal:.2f}%)")
        print("--- END DEBUG ---\n")

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

    def _visualize_binary_rhythm_analysis_for_stage2(self) -> None:
        """
        Pie chart: соотношение 'оставшихся N' к 'не-N' (2 стадия)
        """
        if self.df2_all_annotations is None:
            print("ПРЕДУПРЕЖДЕНИЕ: Данные для анализа stage2 (df2_all_annotations) не созданы. Пропускаем визуализацию.")
            return
        
        normal_count = len(self.df2_all_annotations[
            (self.df2_all_annotations['Type'] == 'N')
        ])

        abnormal_count = len(self.df2_all_annotations) - normal_count

        labels = ['Условно чистая норма (N)', 'Всё остальное (не-N)']
        sizes = [normal_count, abnormal_count]
        colors = ["#1E579D", "#D36907"]

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title("Баланс 'оставшейся нормы' и 'Всего остального (не-N)'")
        plt.axis('equal')  # круглый пирог
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "binary_rhythm_analysis_pie_for_stage2.png"))
        plt.show()

    def _analyze_peak_statistics_for_stage2(self) -> None:
        """
        Выводит статистику по R-пикам (столбец 'Type') для 2-й стадии
        """
        if self.df2_all_annotations is None:
            raise ValueError("self.df2_all_annotations не найден — сначала загрузите данные")
        
        df = self.df2_all_annotations

        print("\n=== Статистика R-пиков (Stage 2) ===")
        print("Всего записей на этапе 2:", len(df))

        # Распределение типов пиков
        type_counts = df['Type'].value_counts()
        type_percent = df['Type'].value_counts(normalize=True).mul(100).round(2)

        # Объединяем в таблицу
        self.peak_statistics_for_stage2 = pd.DataFrame({
            'Count': type_counts,
            'Percent': type_percent
        }).reset_index()
        self.peak_statistics_for_stage2.columns = ['Type', 'Count', 'Percent']

        print("\nРаспределение типов R-пиков:")
        print(self.peak_statistics_for_stage2)

    def _visualize_all_peak_types_for_stage2(self) -> None:
        """
        Визуализирует общую статистику распределению R-пиков на 2й стадии (без N+N)
        """
        if self.peak_statistics_for_stage2.empty:
            print("ПРЕДУПРЕЖДЕНИЕ: Статистика для Stage 2 пуста. Пропускаем график.")
            return
        
        plt.figure(figsize=(14, 6))
        data_for_plot = self.peak_statistics_for_stage2.sort_values(by='Count', ascending=False)
        barplot = sns.barplot(x='Type', y='Count', data=data_for_plot, palette="viridis", hue='Type')
   
        # Подписываем столбцы, используя enumerate для получения числового индекса
        for index, row in enumerate(data_for_plot.itertuples()):
            barplot.text(
                index, 
                row.Count + (0.01 * data_for_plot['Count'].max()),  ## доступ через row.Count
                f"{row.Percent}%",                                  ## доступ через row.Percent
                color='black', 
                ha='center',
                fontsize=10
            )        
        
        plt.title("Распределение всех типов R-пиков")
        plt.xlabel("Тип R-пика")
        plt.ylabel("Количество")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _analyze_r_peak_statistics(self) -> None:
        """
        Проводит анализ R-пиков и формирует датафреймы только с относительными характеристиками:
            - global_peak_distribution: общий профиль (оставлены Count + Percent)
            - patient_peak_top_types: топ-5 типов пиков у каждого пациента (только проценты)
            - patient_normal_abnormal_ratio: доля нормальных / аномальных пиков (только проценты)
            - patient_top_anomaly: самый частый аномальный пик (только процент от аномалий)
        """

        # === 1. Общее распределение типов R-пиков (сохраняем Count + Percent) ===
        total_counts = self.container.df_all_annotations['Type'].value_counts()
        total_percent = (total_counts / total_counts.sum()) * 100

        self.df_global_peak_distribution = pd.DataFrame({
            'Count': total_counts,
            'Percent': total_percent.round(2)
        })

        # === 2. Топ-5 типов пиков по каждому пациенту (только проценты) ===
        patient_peak_top_types = []

        for pid in self.container.df_all_annotations['Patient_id'].unique():
            df_p = self.container.df_all_annotations[self.container.df_all_annotations['Patient_id'] == pid]
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
        for pid in self.container.df_all_annotations['Patient_id'].unique():
            df_p = self.container.df_all_annotations[self.container.df_all_annotations['Patient_id'] == pid]
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
        for pid in self.container.df_all_annotations['Patient_id'].unique():
            df_p = self.container.df_all_annotations[self.container.df_all_annotations['Patient_id'] == pid]
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

    def _visualize_global_peak_distribution(self) -> None:
        df = self.df_global_peak_distribution.sort_values(by='Count', ascending=False)
        
        # Формируем метки для оси X
        new_labels = [f"{peak_type}\n({percent:.1f}%)" for peak_type, percent in zip(df.index, df['Percent'])]

        plt.figure(figsize=(14, 7))
        
        # Строим график по АБСОЛЮТНЫМ значениям (Count)
        barplot = sns.barplot(x=df.index, y=df['Count'], palette="viridis", hue=df.index, dodge=False)

        # Устанавливаем наши новые метки на ось X
        barplot.set_xticklabels(new_labels)

        plt.title("Распределение ВСЕХ типов R-пиков (c % от общего числа)")
        plt.ylabel("Количество")
        plt.xlabel("Тип пика и его доля (в %)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "global_peak_distribution_all.png"))
        plt.show()

    def _visualize_abnormal_peak_ratio(self) -> None:
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

    def _visualize_patient_peak_types_heatmap(self) -> None:
        """
        Heatmap: топ-типы пиков у каждого пациента
        """

        # Формируем матрицу: пациент × топ-5 типов
        df = self.df_patient_peak_top_types.set_index('Patient_id')

        # Берём только типы пиков
        peak_types = df.filter(like='Type_', axis=1)

        # Создаём бинарную матрицу: 1 — тип присутствует у пациента, 0 — нет
        binary_matrix = peak_types.fillna('-').apply(lambda x: x != '-', axis='columns').astype(int)

        plt.figure(figsize=(10, 8))
        sns.heatmap(binary_matrix, cmap='Blues', cbar=False, annot=peak_types.fillna(''), fmt='', linewidths=.5)
        plt.title("Топ-5 типов R-пиков по пациентам (Heatmap)")
        plt.ylabel("Пациент")
        plt.xlabel("Позиция в рейтинге типа")
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "patient_peak_types_heatmap.png"))
        plt.show()

    def _get_color_for_peak_type(self, peak_type: str, mode: str = 'full') -> str:
        """
        Возвращает цвет для типа события.
        
        :param peak_type: Строковое обозначение типа пика (напр., 'N', 'V') или группы ('Normal').
        :param mode: 'full' для цветов по типам, 'reduced' для цветов по группам.
        :return: Строка с HEX-кодом цвета.
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
            '~': '#795548',     # коричневый
            '[': '#795548',     # коричневый
            '!': '#795548',     # коричневый
            ']': '#795548',     # коричневый
            '"': '#795548',     # коричневый
            '?': '#795548',     # коричневый
            'x': '#795548',     # коричневый
            'f': '#795548',     # коричневый
            'other': '#EEEEEE'  # светлый
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

    def _visualize_patient_peak_types_bars(self, mode: str = 'full') -> None:
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
                legend_patches.append(patches.Rectangle((0, 0), 1, 1, color=color, label=t))

        elif mode == 'reduced':
            for t in ['Normal', 'Abnormal', 'BBB', 'Noise']:
                color = self._get_color_for_peak_type(t, mode='reduced')
                legend_patches.append(patches.Rectangle((0, 0), 1, 1, color=color, label=t))
            
        plt.legend(
            handles=legend_patches,
            title="Типы событий",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10
        )

        plt.savefig(os.path.join(self.temp_dir, f"patient_peak_types_bars_{mode}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def _visualize_top_anomalies_pie(self) -> None:
        """
        Pie chart: самые частые аномалии у пациентов
        """

        df = self.df_patient_top_anomaly

        # Считаем, сколько раз какой тип встречается как топ-аномалия
        anomaly_counter = df['Top_anomaly'].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(anomaly_counter, labels=anomaly_counter.index.tolist(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set3"))
        plt.title("Наиболее частые аномалии у пациентов")
        plt.axis('equal')  # круглый пирог
        plt.tight_layout()
        plt.savefig(os.path.join(self.temp_dir, "top_anomalies_pie.png"))
        plt.show()

    def _create_dataframes_for_stage_2(self) -> None:
        """
        Создает новый датафрейм для 2й стадии выполнения задач (без "N+N" R-пиков):
        self.df2_all_annotations
        """
        is_sampe_n = self.container.df_all_annotations['Type'] == 'N'
        is_rhythm_n = self.container.df_all_annotations['Current_Rhythm'] == 'N'
        # Маска всех положительных R-пиков, отсеваемых на 1й стадии
        mask_n_n = is_sampe_n & is_rhythm_n
        
        #Датафрейм df2_all_annotations для анализа для 2й стадии
        self.df2_all_annotations = self.container.df_all_annotations[~mask_n_n]

    def _generate_and_save_detailed_summary(self) -> None:
        """
        Создает и сохраняет детализированную сводку по R-пикам для каждого пациента в JSON-файл.
        Сводка включает распределение по итоговым классам Stage 2 и детализацию по сырым типам.
        """
        print("Начинаю генерацию детализированной сводки по пациентам...")
        
        # Карта для группировки сырых типов в классы Stage 2
        GROUP_MAP = {
            'A': 'subSVEB', 'a': 'subSVEB', 'J': 'subSVEB', 'e': 'subSVEB', 'j': 'subSVEB',
            'V': 'VEB', 'E': 'VEB',
            'F': 'Fusion', '+': 'Fusion',
            'Q': 'Q', '/': 'Q', '!': 'Q', '~': 'Q', 'f': 'Q', 'U': 'Q', '?': 'Q', '"': 'Q', 'x': 'Q', '[': 'Q', ']': 'Q',
            'N': 'N (по Aux не N)', 
            'L': 'L',
            'R': 'R'
        }

        # Используем df_all_annotations, так как нужна полная картина
        df_annos = self.container.df_all_annotations
        all_patients_summary = {}

        for pid in tqdm(df_annos['Patient_id'].unique(), desc="Анализ пациентов для сводки"):
            df_patient = df_annos[df_annos['Patient_id'] == pid]
            total_peaks = len(df_patient)
            if total_peaks == 0:
                continue

            # Считаем сырые типы
            raw_type_counts = df_patient['Type'].value_counts().to_dict()
            
            # Считаем N+N отдельно
            n_plus_n_count = len(df_patient[(df_patient['Type'] == 'N') & (df_patient['Current_Rhythm'] == 'N')])
            
            # Формируем итоговую структуру
            patient_distribution = {}
            
            # Добавляем N+N в распределение, если они есть
            if n_plus_n_count > 0:
                patient_distribution['N+N'] = {
                    "total_percent": round((n_plus_n_count / total_peaks) * 100, 2),
                    "details": {"N": round((n_plus_n_count / total_peaks) * 100, 2)}
                }

            # Создаем временный словарь для подсчета групп
            group_counts = defaultdict(int)
            for raw_type, count in raw_type_counts.items():
                group = GROUP_MAP.get(raw_type)
                if group:
                    # Особая логика для 'N': считаем только те, что НЕ 'N+N'
                    if raw_type == 'N':
                        # Считаем 'N' пики, где ритм НЕ 'N'
                        n_not_n_rhythm_count = len(df_patient[(df_patient['Type'] == 'N') & (df_patient['Current_Rhythm'] != 'N')])
                        if n_not_n_rhythm_count > 0:
                           group_counts[group] += n_not_n_rhythm_count
                    else:
                        group_counts[group] += count
            
            # Собираем финальный JSON-объект
            for group, group_total_count in group_counts.items():
                patient_distribution[group] = {
                    "total_percent": round((group_total_count / total_peaks) * 100, 2),
                    "details": {}
                }
                # Находим все сырые типы для этой группы
                for raw_type, mapped_group in GROUP_MAP.items():
                    if mapped_group == group and raw_type in raw_type_counts:
                        raw_count = raw_type_counts[raw_type]
                        # Особая логика для 'N'
                        if raw_type == 'N':
                            raw_count = len(df_patient[(df_patient['Type'] == 'N') & (df_patient['Current_Rhythm'] != 'N')])

                        if raw_count > 0:
                            patient_distribution[group]['details'][raw_type] = round((raw_count / total_peaks) * 100, 2)

            all_patients_summary[str(pid)] = {
                "total_peaks": total_peaks,
                "distribution": patient_distribution
            }

        # Сохраняем результат
        summary_path = os.path.join(config['paths']['data_dir'], "patient_detailed_summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(all_patients_summary, f, ensure_ascii=False, indent=4)
            print(f"Детализированная сводка по пациентам успешно сохранена в: {summary_path}")
        except Exception as e:
            print(f"ОШИБКА при сохранении детализированной сводки: {e}")

    def _analyze_class_distribution_by_patient(self) -> None:
            """
            Анализирует и выводит распределение итоговых классов Stage 2 по пациентам.
            Отчет имеет вид: Класс -> Пациент -> Количество пиков.
            Эта информация используется для принятия решения о стратифицированном разделении выборки.
            """
            print("\n" + "="*20 + " Анализ распределения классов по пациентам " + "="*20)
            
            df_annos = self.container.df_all_annotations.copy()

            # Карта для группировки сырых типов
            GROUP_MAP = {
                'A': 'subSVEB', 'a': 'subSVEB', 'J': 'subSVEB', 'e': 'subSVEB', 'j': 'subSVEB',
                'V': 'VEB', 'E': 'VEB',
                'F': 'Fusion', '+': 'Fusion',
                'Q': 'Q', '/': 'Q', '!': 'Q', '~': 'Q', 'f': 'Q', 'U': 'Q', '?': 'Q', '"': 'Q', 'x': 'Q', '[': 'Q', ']': 'Q',
                'N': 'N (по Aux не N)', 
                'L': 'L',
                'R': 'R'
            }
            
            # Создаем временную колонку с итоговым классом
            def get_final_class(row):
                if row['Type'] == 'N' and row['Current_Rhythm'] == 'N':
                    return 'N+N'
                return GROUP_MAP.get(row['Type'], 'Unknown')

            df_annos['Final_Class'] = df_annos.apply(get_final_class, axis=1)

            # Группируем и считаем
            class_patient_counts = df_annos.groupby('Final_Class')['Patient_id'].value_counts()

            # Выводим отчет
            for final_class in sorted(class_patient_counts.index.get_level_values(0).unique()):
                print(f"\n--- Класс: {final_class} ---")
                
                # Фильтруем данные для текущего класса и сортируем по убыванию количества
                class_data = class_patient_counts[final_class].sort_values(ascending=False)
                
                # Преобразуем в более читаемый формат для вывода
                output_lines = []
                for patient_id, count in class_data.items():
                    output_lines.append(f"  - Пациент {patient_id}: {count} пиков")
                
                print("\n".join(output_lines))
            
            print("="*70)


    def analyze_split_balance(self) -> None:
        """
        Анализирует и выводит баланс классов для предопределенных train/val/test выборок.
        Использует списки пациентов из config.yaml и DataFrame с профилями,
        который генерируется на лету.
        """
        print("\n" + "="*25 + " Анализ баланса классов в выборках " + "="*25)

        # 1. Загружаем списки пациентов из конфига
        test_pids_config = config['data']['patient_split']['test_pids']
        val_pids_config = config['data']['patient_split']['val_pids']

        # 2. Подготавливаем DataFrame с профилями (аналогично create_patient_profiles_table)
        df_annos = self.container.df_all_annotations.copy()
        unmapped_types = set() # Сюда будем собирать все неопознанные символы

        GROUP_MAP = {
            # subSVEB
            'A': 'subSVEB', 'a': 'subSVEB', 'J': 'subSVEB', 'e': 'subSVEB', 'j': 'subSVEB', 'S': 'subSVEB',
            # VEB
            'V': 'VEB', 'E': 'VEB',
            # Fusion
            'F': 'Fusion', '+': 'Fusion',
            # Q (включая все шумы и неопределенные)
            'Q': 'Q', '/': 'Q', '!': 'Q', '~': 'Q', 'f': 'Q', 'U': 'Q', '?': 'Q', '"': 'Q', 'x': 'Q', '[': 'Q', ']': 'Q', '|': 'Q',
            # N-
            'N': 'N-',
            # BBB
            'L': 'L',
            'R': 'R'
        }

        def get_final_class(row):
            peak_type = row['Type']
            
            if peak_type == 'N' and row['Current_Rhythm'] == 'N':
                return 'N+'
            
            final_class = GROUP_MAP.get(peak_type)
            if final_class is None:
                unmapped_types.add(peak_type) # Нашли неопознанный тип - добавляем в set
                return 'Unknown'
            return final_class


###        def get_final_class(row):
###            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N': return 'N+'
###            # Для редких/неизвестных типов, не попавших в карту
###            if row['Type'] not in GROUP_MAP: return 'Unknown'
###            return GROUP_MAP.get(row['Type'])

        df_annos['Final_Class'] = df_annos.apply(get_final_class, axis=1)

        # 3. ДИАГНОСТИЧЕСКИЙ ВЫВОД: показываем, что попало в 'Unknown'
        if unmapped_types:
            print("\n" + "!"*20 + " ДИАГНОСТИКА 'UNKNOWN' " + "!"*20)
            print(f"Следующие типы R-пиков не найдены в GROUP_MAP и были отнесены к классу 'Unknown':")
            print(f" -> {sorted(list(unmapped_types))}")
            print("!"*65)

        # Создаем сводную таблицу (pivot table)
        df_profiles = pd.pivot_table(
            df_annos,
            values='Sample',
            index='Patient_id',
            columns='Final_Class',
            aggfunc='count',
            fill_value=0
        )
        
        # Добавляем итоговые столбцы
        all_stage2_classes = [cls for cls in df_profiles.columns if cls != 'N+']

        df_profiles['Total_Stage2'] = df_profiles[all_stage2_classes].sum(axis=1)
        #df_profiles['Total_Peaks'] = df_profiles.sum(axis=1)

        # Теперь Total_Peaks - это просто N+ и Total_Stage2. Больше нет двойного счета.
        df_profiles['Total_Peaks'] = df_profiles.get('N+', 0) + df_profiles['Total_Stage2']


        # 3. Определяем все группы пациентов
        all_pids_in_data = set(df_profiles.index.astype(str))
        test_pids_set = set(map(str, test_pids_config))
        val_pids_set = set(map(str, val_pids_config))
        
        # Пациенты для train - это все, кто не в test и не в val
        train_pids_set = all_pids_in_data - test_pids_set - val_pids_set

        splits = {
            'Train': sorted(list(train_pids_set)),
            'Validation': sorted(list(val_pids_set)),
            'Test': sorted(list(test_pids_set))
        }

        # 4. Собираем статистику
        results = {}
        all_classes_sorted = ['N+'] + sorted([col for col in df_profiles.columns if col not in ['Total_Peaks', 'Total_Stage2', 'N+']])

        for split_name, pids in splits.items():
            # Проверяем, есть ли пациенты из списка в данных
            existing_pids = [pid for pid in pids if pid in df_profiles.index]
            if not existing_pids:
                df_split = pd.DataFrame(columns=df_profiles.columns).fillna(0)
            else:
                df_split = df_profiles.loc[existing_pids]
            
            counts = df_split.sum()
            total_peaks = counts.get('Total_Peaks', 0)
            total_stage2 = counts.get('Total_Stage2', 0)
            
            results[split_name] = {
                'counts': counts,
                'total_peaks': total_peaks,
                'total_stage2': total_stage2,
                'num_patients': len(existing_pids)
            }

        # 5. Формируем таблицу "Полное распределение"
        table1_data = []
        headers1 = [
            "Класс", 
            f"Train ({results['Train']['num_patients']} пац.)", 
            f"Val ({results['Validation']['num_patients']} пац.)", 
            f"Test ({results['Test']['num_patients']} пац.)"
        ]
        
        for class_name in all_classes_sorted:
            row = [class_name]
            for split_name in ['Train', 'Validation', 'Test']:
                count = results[split_name]['counts'].get(class_name, 0)
                
                # Всегда считаем процент от общего числа пиков
                total_peaks = results[split_name]['total_peaks']

                percent = (count / total_peaks * 100) if total_peaks > 0 else 0
                row.append(f"{int(count)} ({percent:.1f}%)")
            table1_data.append(row)

        print("\n" + "="*30 + " Полное распределение классов по выборкам " + "="*30)
        print(tabulate(table1_data, headers=headers1, tablefmt="grid"))

        # 6. Формируем таблицу "Баланс для Stage 1"
        table2_data = []
        headers2 = ["Выборка", "N+ (Норма)", "Остальные (Аномалии)", "Всего пиков", "% N+ из всех"]

        for split_name in ['Train', 'Validation', 'Test']:
            n_plus_count = results[split_name]['counts'].get('N+', 0)
            others_count = results[split_name]['total_peaks'] - n_plus_count
            total_count = results[split_name]['total_peaks']

            percent_normal = (n_plus_count / total_count * 100) if total_count > 0 else 0
            row = [
                split_name,
                f"{int(n_plus_count)}",
                f"{int(others_count)}",
                f"{int(total_count)}",
                f"{percent_normal:.1f}%"
            ]
            table2_data.append(row)
            
        print("\n" + "="*30 + " Баланс классов для Stage 1 " + "="*30)
        print(tabulate(table2_data, headers=headers2, tablefmt="grid"))
        print("="*85)

    def create_patient_profiles_table(self) -> None:
        """
        Создает, выводит в консоль и сохраняет в CSV сводную таблицу-профиль по каждому пациенту.
        Таблица содержит количество пиков каждого итогового класса и используется для
        принятия решения о стратифицированном разделении.

        В консольном выводе "занятые" пациенты из config['data']['patient_busy']
        отображаются отдельно от "свободных".
        """
        print("\n" + "="*20 + " Генерация профилей пациентов для стратификации " + "="*20)
        
        df_annos = self.container.df_all_annotations.copy()

        # Карта для группировки сырых типов
        GROUP_MAP = {
            'A': 'subSVEB', 'a': 'subSVEB', 'J': 'subSVEB', 'e': 'subSVEB', 'j': 'subSVEB',
            'V': 'VEB', 'E': 'VEB', 'F': 'Fusion', '+': 'Fusion',
            'Q': 'Q', '/': 'Q', '!': 'Q', '~': 'Q', 'f': 'Q', 'U': 'Q', '?': 'Q', '"': 'Q', 'x': 'Q', '[': 'Q', ']': 'Q',
            'N': 'N-', 'L': 'L', 'R': 'R'
        }
        
        def get_final_class(row):
            if row['Type'] == 'N' and row['Current_Rhythm'] == 'N': return 'N+'
            return GROUP_MAP.get(row['Type'], 'Unknown')

        df_annos['Final_Class'] = df_annos.apply(get_final_class, axis=1)

        # Создаем сводную таблицу (pivot table)
        profiles_df = pd.pivot_table(
            df_annos, 
            values='Sample', 
            index='Patient_id', 
            columns='Final_Class', 
            aggfunc='count', 
            fill_value=0
        )

        # Добавляем итоговые столбцы
        all_stage2_classes = [cls for cls in profiles_df.columns if cls != 'N+']
        ### profiles_df['Total_Stage2'] = profiles_df[all_stage2_classes].sum(axis=1)
        ### profiles_df['Total_Peaks'] = profiles_df.sum(axis=1)
        if 'Total_Stage2' not in profiles_df.columns:
            profiles_df['Total_Stage2'] = profiles_df[all_stage2_classes].sum(axis=1)
        if 'Total_Peaks' not in profiles_df.columns:
            profiles_df['Total_Peaks'] = profiles_df.sum(axis=1) - profiles_df['Total_Stage2'] # Исправлено, чтобы избежать двойного счета


        ### # Сортируем столбцы для наглядности
        ### sorted_columns = ['Total_Peaks', 'Total_Stage2', 'N+', 'N-'] + sorted([c for c in all_stage2_classes if c != 'N-'])
        ### profiles_df = profiles_df[sorted_columns]

        # Сортируем столбцы для наглядности
        # Убедимся, что все ожидаемые колонки существуют перед сортировкой
        existing_cols = profiles_df.columns.tolist()
        base_cols = ['Total_Peaks', 'Total_Stage2', 'N+', 'N-']
        other_cols = sorted([c for c in existing_cols if c not in base_cols])
        sorted_columns = [col for col in base_cols if col in existing_cols] + other_cols
        profiles_df = profiles_df[sorted_columns]


        # Разделение на "занятых" и "свободных" пациентов
        # 1. Загружаем списки зарезервированных пациентов из конфига
        busy_config = config['data'].get('patient_busy', {})
        busy_test = set(map(str, busy_config.get('test_pids', [])))
        busy_val = set(map(str, busy_config.get('val_pids', [])))
        busy_train = set(map(str, busy_config.get('train_pids', []))) # Используем 'train_pids'
        
        all_busy_pids = busy_test.union(busy_val, busy_train)

        # 2. Получаем списки ID из нашего DataFrame
        all_pids_in_df = profiles_df.index.astype(str).tolist()

        # 3. Разделяем ID на две группы
        busy_pids_in_df = sorted([pid for pid in all_pids_in_df if pid in all_busy_pids])
        free_pids_in_df = sorted([pid for pid in all_pids_in_df if pid not in all_busy_pids])
        
        # 4. Создаем два отдельных DataFrame для вывода
        df_busy = profiles_df.loc[busy_pids_in_df] if busy_pids_in_df else pd.DataFrame()
        df_free = profiles_df.loc[free_pids_in_df] if free_pids_in_df else pd.DataFrame()

        print("Сводная таблица профилей пациентов:")
        # 5. Выводим сначала "занятых"
        if not df_busy.empty:
            print("\n" + "--- Зарезервированные (busy) пациенты ---".center(80))
            print(df_busy.to_string())
        
        # 6. Выводим разделитель, если есть обе группы
        if not df_busy.empty and not df_free.empty:
            # Создаем разделитель по ширине таблицы
            header_line = df_busy.to_string().split('\n')[0]
            print('-' * len(header_line))

        # 7. Выводим "свободных"
        if not df_free.empty:
            print("\n" + "--- Свободные (free) пациенты ---".center(80))
            print(df_free.to_string())


        ### print(profiles_df.to_string())

        # Сохраняем в CSV для удобства
        profiles_path = os.path.join(config['paths']['data_dir'], "patient_profiles.csv")
        profiles_df.to_csv(profiles_path)
        print(f"\nТаблица профилей сохранена в: {profiles_path}")
        print("="*80)