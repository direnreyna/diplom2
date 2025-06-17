# dataset_loading

import os
import pandas as pd

from tqdm import tqdm
from config import config
from typing import Tuple, List

class DatasetLoading:
    def __init__(self) -> None:
        self.temp_dir = config['paths']['temp_dir']
        self.patient_ids = []
        self.df_all_signals = pd.DataFrame()
        self.df_all_annotations = pd.DataFrame()
    
    def pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        Формирует датафреймы:
        self.df_all_signals     - с подробными данными ЭКГ по всем пациентам
        self.df_all_annotations - с расшифровкой сигналов по всем пациентам 
        """

        self.patient_ids = [os.path.basename(f)[:3] for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        all_signals = []
        all_annotations = []

        for pid in tqdm(self.patient_ids, desc="Собираем информацию по пациентам", unit=" пациент"):
            # Загрузка CSV сигналов в список датафреймов. 'Sample', 'MLII', 'V1', 'Patient_id'
            df_signal = pd.read_csv(os.path.join(self.temp_dir, f'{pid}.csv')).rename(columns={"'sample #'": 'Sample'})##[["Sample", "'MLII'"]]
            #print("Колонки df_signal: ", pid, list(df_signal.columns))

            df_signal['Patient_id'] = pid
            df_signal.columns = [col.strip("'") for col in df_signal.columns]   ## Убираем кавычки из названия колонок
            all_signals.append(df_signal)

            # Загрузка TXT аннотаций в список датафреймов. Столбцы 'Sample', 'Type', 'Patient_id'
            df_annotation = pd.read_csv(
                os.path.join(self.temp_dir, f'{pid}annotations.txt'),
                sep=r'\s+',
                skiprows=1,
                header=None,
                names=['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num', 'Aux'],
                quoting=3
            )[['Sample', 'Type', 'Aux']]                               ## Оставляем только значимые столбцы: маркер и метку
            df_annotation['Patient_id'] = pid
            all_annotations.append(df_annotation)

        # Объединённые датафреймы
        self.df_all_signals = pd.concat(all_signals, ignore_index=True)
        self.df_all_annotations = pd.concat(all_annotations, ignore_index=True)

        # Преобразуем Sample в int
        self.df_all_signals['Sample'] = self.df_all_signals['Sample'].astype(int)
        self.df_all_annotations['Sample'] = self.df_all_annotations['Sample'].astype(int)

        # Создаем колонку Label, используя маппинг "тип метки" -> целое число
        label_map = {t: i for i, t in enumerate(self.df_all_annotations['Type'].unique())}
        self.df_all_annotations['Label'] = self.df_all_annotations['Type'].map(label_map)

        print("=== df_all_signals ===")
        print(self.df_all_signals.head())
        print("\nКолонки:", list(self.df_all_signals.columns))
        print("\nРазмер:", len(self.df_all_signals))

        print("\n=== df_all_annotations ===")
        print(self.df_all_annotations.head())
        print("\nКолонки:", list(self.df_all_annotations.columns))
        print("\nТипы аннотаций:", self.df_all_annotations['Type'].unique())

        return self.df_all_signals, self.df_all_annotations, self.patient_ids
    
    def check_datasets_exists(self):
        """
        Проверяет наличие сохраненных ранее датасетов на диске

        :return:
            - True/False - если есть все 4 файла
        """
        exists_all = True
        dataset_path = config['paths']['data_dir']
        prefixes = ['top', 'cross', 'uni1', 'uni2']
        # stages = ['stage1', 'stage2']
        stages = ['stage1']
        
        dataset_name = config['data']['dataset_name']
        for pr in prefixes:
            for stage in stages:
                file_dataset = os.path.join(dataset_path, f"{pr}_{stage}_{dataset_name}")
                print(f"Проверяю наличие файла: [{file_dataset}]")
                if not os.path.exists(file_dataset):
                    exists_all = False
                    break
                else:
                    print(f"Обнаружен файл: {file_dataset}")
        print(f"Пропускаем сбор датасетов, переходим к обучению...")
        return exists_all
