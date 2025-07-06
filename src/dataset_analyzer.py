# src/dataset_analyzer.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from config import config

class DatasetAnalyze:
    def __init__(self, data_container):
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

            # Добавление в аннотации ритмов из Aux
        self._add_rhythm_annotations()                                          # Добавляет колонку Current_Rhythm в df_all_annotations на основе колонки Aux
        if self.check_analytics:
            self._analyze_Current_Rhythm_statistics(self.container.df_all_annotations)    # Формирует общую статистику по Aux-событиям
            self._analyze_patient_rhythm_type_stats()                           # Формирует статистику Aux-событий по каждоиу пациенту
            self._binary_rhythm_type_analysis()                                 # Анализирует баланс нормальных R-пиков в нормальных Aux-событиях
            self._visualize_global_rhythm_distribution()                        # 
            self._visualize_rhythm_abnormal_distribution()                      # 
            self._visualize_binary_rhythm_analysis()                            # Визуализирует баланс нормальных R-пиков в нормальных Aux-событиях

            # Анализ R-пиков и типов событий
            self._analyze_r_peak_statistics()                                   # Формирует статистику по R-пикам и типам событий 
            self._visualize_global_peak_distribution()                          # Pie / barplot: общее распределение типов R-пиков
            self._visualize_abnormal_peak_ratio()                               # Barplot: процент аномальных пиков на пациента
            self._visualize_patient_peak_types_heatmap()                        # Heatmap: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='full')                # Color-bars: топ-типы пиков у каждого пациента
            self._visualize_patient_peak_types_bars(mode='reduced')             # Color-bars: аггрегированные топ-типы пиков у каждого пациента
            self._visualize_top_anomalies_pie()                                 # Pie chart: самые частые аномалии по пациентам
        
        # 2я стадия выделение датафреймов без N+N R-пиков
        self._create_dataframes_for_stage_2()                                   # создает self.df2_all_signals и self.df2_all_annotations

        if self.check_analytics:
            # Анализ R-пиков и типов событий
            self._analyze_Current_Rhythm_statistics(self.df2_all_annotations)   # Формирует общую статистику по Aux-событиям
            self._binary_rhythm_type_analysis_for_stage2(self.df2_all_annotations)  # Анализирует баланс нормальных R-пиков вне нормальных Aux-событиий (стадия 2)
            self._visualize_global_rhythm_distribution(stage='stage_2_')        # 
            self._visualize_binary_rhythm_analysis_for_stage2()                 # 
            self._analyze_peak_statistics_for_stage2()                          # Формирует общую статистику распределению R-пиков на 2й стадии (без N+N)
            self._visualize_all_peak_types_for_stage2()                         # Визуализирует общую статистику распределению R-пиков на 2й стадии (без N+N)

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

    def _visualize_channel_distribution(self):
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

    def _analyze_Current_Rhythm_statistics(self, df_all_annotations) -> None:
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

    def _binary_rhythm_type_analysis_for_stage2(self, df_all_annotations) -> None:
        """
        Проводит бинарный анализ для 2й стадии:
            - Категория 0: Type == 'N' → "почти чистая норма"
            - Категория 1: все остальные случаи → "аномалия"
        
        Выводит статистику по всему датасету.
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

    def _visualize_global_rhythm_distribution(self, stage='') -> None:
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

        normal_count = len(self.container.df_all_annotations[
            (self.container.df_all_annotations['Type'] == 'N') &
            (self.container.df_all_annotations['Current_Rhythm'] == 'N')
        ])

        abnormal_count = len(self.container.df_all_annotations) - normal_count

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