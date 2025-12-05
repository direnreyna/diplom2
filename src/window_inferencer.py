# src/window_inferencer

import os
import json
import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt

from .config import config
from .self_attention_block import SelfAttentionBlock
from .model_trainer import CategoricalFocalLoss, MacroF1Score

from typing import Tuple, cast, Dict, Any
from tensorflow.keras.models import Model, load_model

class WindowInference:
    """
    Центральный класс для выполнения каскадного инференса.
    Загружает все необходимые артефакты при инициализации,
    учитывая настройки в config.yaml (load_from_mlflow).
    """
    def __init__(self, prefix='top'):
        """
        Инициализирует пайплайн инференса.

        Загружает все необходимые артефакты:
        - Модели для stage1 и stage2.
        - Тестовые данные и метаданные для обеих стадий.
        - Детализированную сводку по пациентам из JSON-файла.
        - Формирует структуры для быстрого поиска и отображения в GUI.

        :param prefix: Префикс датасета ('top', 'cross' и т.д.), для которого будут загружены модели и данные.
        """
        print("Инициализация WindowInference...")
        self.prefix = prefix
        self.config = config
        
        print("\n" + "="*50)
        print("ЗАПУСК ТЕСТИРОВАНИЯ НОВОГО КАСКАДНОГО ИНФЕРЕНСА")
        print("="*50)

        # Загружаем модели
        self.model_stage1 = self._load_model_for_stage('stage1')
        self.model_stage2 = self._load_model_for_stage('stage2')
        print("Модели для stage1 и stage2 успешно загружены.")

###        # Загружаем данные и метаданные
###        self.data_stage1 = self._load_data_for_stage('stage1')
###        self.data_stage2 = self._load_data_for_stage('stage2')
###        print("Данные и метаданные для stage1 и stage2 успешно загружены.")
###
###        # Загружаем полный размеченный массив метаданных
###        self.metadata_labeled_s1 = self.data_stage1['metadata_labeled']
###        self.metadata_labeled_s2 = self.data_stage2['metadata_labeled']
###        
###        # Создаем обратный индекс для получения статуса пика (train/val/test)
###        self.split_status_index = { (meta[0], meta[1]): meta[2] for meta in self.metadata_labeled_s1 }
###
###        # Создаем индексы для быстрого поиска
###        self.metadata_index_s1 = self._create_metadata_index(self.data_stage1['metadata_test'])
###        self.metadata_index_s2 = self._create_metadata_index(self.data_stage2['metadata_test'])
###        print("Индексы для быстрого поиска по метаданным созданы.")

        # Загружаем данные для разных задач
        # Только ТЕСТОВЫЕ данные для функции "Случайный пример"
        self.data_stage1_test_only = self._load_test_data_for_stage('stage1')
        print("Тестовые данные для случайного инференса загружены.")

        # ВСЕ данные (train+val+test) для функции "Анализ по ID"
        self.data_stage1_full = self._load_data_for_stage('stage1')
        self.data_stage2_full = self._load_data_for_stage('stage2')
        print("Полные данные для инференса по ID загружены.")

        # --- Создаем индексы для быстрого поиска ---
        # Индекс по ВСЕМ данным для поиска по ID
        self.metadata_index_s1 = self._create_metadata_index(self.data_stage1_full['metadata'])
        self.metadata_index_s2 = self._create_metadata_index(self.data_stage2_full['metadata'])
        print("Индексы для быстрого поиска по всем метаданным созданы.")
        
        # --- Вспомогательные структуры ---
        # Загружаем полный размеченный массив метаданных для GUI
        self.metadata_labeled_s1 = self.data_stage1_full['metadata_labeled']

        # Загружаем метки классов
        self.class_labels_s1 = self.config['class_labels']['stage1']
        self.class_labels_s2 = self.config['class_labels']['stage2']
        self.display_names = self.config['display_names']

        # Загружаем детализированную сводку по пациентам для "умного" dropdown и статистики
        self.patient_summary = self._load_patient_summary()
        self.formatted_patient_list = self._create_formatted_patient_list()

        print("InferencePipeline готов к работе.")

    def _load_model_for_stage(self, stage: str) -> Model:
        """Загружает модель для указанной стадии, соблюдая логику `load_from_mlflow` из config.yaml."""
        load_from_mlflow = self.config['execution']['load_from_mlflow']
        # Объекты, необходимые для загрузки любой из моделей
        custom_objects = {
            'SelfAttentionBlock': SelfAttentionBlock,
            'CategoricalFocalLoss': CategoricalFocalLoss,
            'MacroF1Score': MacroF1Score                            ## Используем класс метрики вместо функции
            # 'f1_score': f1_score
        }
        model = None

        # СЦЕНАРИЙ 1: Загрузка из MLflow
        if load_from_mlflow:
            run_id = self.config['execution']['mlflow_run_id'].get(stage)
            if not run_id:
                raise ValueError(f"load_from_mlflow=True, но mlflow_run_id для стадии '{stage}' не найден в config.yaml")
            
            print(f"Загрузка модели для стадии '{stage}' из MLflow run_id: {run_id}")
            try:
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.keras.load_model(model_uri, custom_objects=custom_objects)
                print(f"Модель для стадии '{stage}' из MLflow успешно загружена.")
            except Exception as e:
                raise RuntimeError(f"Не удалось загрузить модель из MLflow для run_id {run_id}. Ошибка: {e}")
        # СЦЕНАРИЙ 2: Загрузка с диска
        else:
            model_path = os.path.join(
                self.config['paths']['model_dir'],
                f"{self.prefix}_{stage}_{self.config['paths']['best_model']}"
            )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Модель не найдена по пути: {model_path}. Убедитесь, что модель обучена и сохранена.")
            
            print(f"Загрузка модели для стадии '{stage}' с диска: {model_path}")
            # Используем tf.keras.models.load_model для локальных файлов
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"Модель для стадии '{stage}' с диска успешно загружена.")

        if model is None:
            raise RuntimeError(f"Не удалось загрузить модель для стадии '{stage}' ни одним из способов.")

        return cast(Model, model)

    def _load_data_for_stage(self, stage: str) -> Dict[str, np.ndarray]:
        """Вспомогательный метод для загрузки ВСЕХ данных для указанной стадии."""
        dataset_path = os.path.join(
            self.config['paths']['data_dir'],
            f"{self.prefix}_{stage}_{self.config['data']['dataset_name']}"
        )
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}.")
            
        with np.load(dataset_path, allow_pickle=True) as data:
            # Загружаем все части датасета
            X = np.concatenate([data['X_train'], data['X_val'], data['X_test']], axis=0)
            y = np.concatenate([data['y_train'], data['y_val'], data['y_test']], axis=0)
            metadata = np.concatenate([data['metadata_train'], data['metadata_val'], data['metadata_test']], axis=0)
            
            return {
                'X': X,
                'y': y,
                'metadata': metadata,
                'metadata_labeled': data['metadata_labeled']
            }
        
    def _load_test_data_for_stage(self, stage: str) -> Dict[str, np.ndarray]:
        """Вспомогательный метод для загрузки ТОЛЬКО ТЕСТОВЫХ данных для указанной стадии."""
        dataset_path = os.path.join(
            self.config['paths']['data_dir'],
            f"{self.prefix}_{stage}_{self.config['data']['dataset_name']}"
        )
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}.")
            
        with np.load(dataset_path, allow_pickle=True) as data:
            return {
                'X_test': data['X_test'],
                'y_test': data['y_test'],
                'metadata_test': data['metadata_test'],
            }
           
    def _create_metadata_index(self, metadata_array: np.ndarray) -> Dict[Tuple, int]:
        """Создает словарь для быстрого поиска индекса в массиве по метаданным."""
        return {tuple(meta): i for i, meta in enumerate(metadata_array)}

    def predict_random(self) -> Dict[str, Any]:
        """
        Выбирает случайный R-пик из тестовой выборки stage1, выполняет
        полный каскадный инференс и НЕ визуализирует результат (это задача GUI).
        """
        print("\nЗапуск инференса для случайного R-пика.")
        
        # 1. Выбираем случайный индекс из тестовой выборки stage1
        num_test_samples = len(self.data_stage1_test_only['X_test'])        
        random_index = np.random.randint(0, num_test_samples)
        
        # 2. Получаем "паспорт" этого случайного пика
        metadata_to_find = tuple(self.data_stage1_test_only['metadata_test'][random_index])
        patient_id, sample_id = metadata_to_find
        
        print(f"Выбран случайный R-пик: Patient ID = {patient_id}, Sample ID = {int(sample_id)}")
        
        # 3. Вызываем основной метод, который вернет полный словарь с результатами
        results = self.predict_by_id(patient_id, int(sample_id))

        # Печатаем результаты в консоль для логгирования
        print("\nРезультаты каскадного инференса:")
        true_label_s1_display = self.display_names.get(results['true_label_s1'], results['true_label_s1'])
        pred_label_s1_display = self.display_names.get(results['prediction_s1'], results['prediction_s1'])
        print(f"Истинная метка Stage 1: {true_label_s1_display}")
        print(f"Предсказание Stage 1: {pred_label_s1_display} (Уверенность: {results['confidence_s1']:.2f}%)")
        ### print(f"Истинная метка Stage 1: {results['true_label_s1']}")
        ### print(f"Предсказание Stage 1: {results['prediction_s1']} (Уверенность: {results['confidence_s1']:.2f}%)")
        
        title = f"Пациент: {results['patient_id']}, Сэмпл: {results['sample_id']}\n"
        title += f"Предсказание S1: {results['prediction_s1']}"

        if 'prediction_s2' in results:
            true_label_s2_display = self.display_names.get(results['true_label_s2'], results['true_label_s2'])
            pred_label_s2_display = self.display_names.get(results['prediction_s2'], results['prediction_s2'])
            print(f"Истинная метка Stage 2: {true_label_s2_display}")
            print(f"Предсказание Stage 2: {pred_label_s2_display} (Уверенность: {results['confidence_s2']:.2f}%)")
            ### print(f"Истинная метка Stage 2: {results['true_label_s2']}")
            ### print(f"Предсказание Stage 2: {results['prediction_s2']} (Уверенность: {results['confidence_s2']:.2f}%)")
            ### title += f" -> S2: {results['prediction_s2']}"

            common_result = results['confidence_s1'] * results['confidence_s2'] / 100
            print(f"Общая уверенность: {common_result:.2f}%")
            ### print(f"Уверенность по 2 стадиям: {common_result:.2f}%)")
        else:
            print("Инференс Stage 2 не проводился.")

        # Вызываем визуализацию, передавая данные окна и сформированный заголовок
        ### self.visualize_peak(results['window_data'], title=title)
            
        return results

    def predict_by_id(self, patient_id: str, sample_id: int) -> Dict[str, Any]:
        """
        Выполняет полный каскадный инференс для R-пика по его "паспорту" (ID).
        
        :param patient_id: ID пациента (например, '101').
        :param sample_id: ID сэмпла (отсчета), в котором находится R-пик.
        :return: Словарь с подробными результатами инференса.
        """
        peak_id = (str(patient_id), sample_id) # Гарантируем строковый тип ID пациента
        results = {'patient_id': patient_id, 'sample_id': sample_id}

        #################################################################################
        # Этап 1: Предсказание на модели Stage 1
        #################################################################################
        if peak_id not in self.metadata_index_s1:
            raise ValueError(f"R-пик с ID {peak_id} не найден в данных stage1.")
                    
        idx_s1 = self.metadata_index_s1[peak_id]
        window_s1 = self.data_stage1_full['X'][idx_s1]
        true_label_ohe_s1 = self.data_stage1_full['y'][idx_s1]
        true_label_idx_s1 = int(true_label_ohe_s1) # Для бинарного случая

        # Добавляем batch-измерение для предсказания
        prediction_s1_raw = self.model_stage1.predict(np.expand_dims(window_s1, axis=0))[0]
        
        pred_label_idx_s1 = int(prediction_s1_raw[0] > 0.5)
        # Рассчитываем уверенность в предсказанном классе
        if pred_label_idx_s1 == 1: # Если предсказан 'Alert'
            confidence_s1 = prediction_s1_raw[0] * 100
        else: # Если предсказан 'Good'
            confidence_s1 = (1 - prediction_s1_raw[0]) * 100

        results.update({
            'window_data': window_s1,
            'true_label_s1': self.class_labels_s1[true_label_idx_s1],
            'prediction_s1': self.class_labels_s1[pred_label_idx_s1],
            'confidence_s1': confidence_s1
        })
        
        #################################################################################
        # Этап 2: Условное предсказание на модели Stage 2
        #################################################################################
        # Продолжаем, только если предсказание stage1 было "Alert"
        if pred_label_idx_s1 == 1:
            if peak_id not in self.metadata_index_s2:
                print(f"Предупреждение: Пик {peak_id} предсказан как 'Alert', но не найден в данных stage2.")
                return results

            idx_s2 = self.metadata_index_s2[peak_id]
            # Важно: для stage2 мы используем данные из датасета stage2, но можем визуализировать окно из stage1
            window_s2 = self.data_stage2_full['X'][idx_s2]
            true_label_ohe_s2 = self.data_stage2_full['y'][idx_s2]
            true_label_idx_s2 = np.argmax(true_label_ohe_s2)

            prediction_s2_raw = self.model_stage2.predict(np.expand_dims(window_s2, axis=0))[0]
            pred_label_idx_s2 = np.argmax(prediction_s2_raw)
            
            results.update({
                'true_label_s2': self.class_labels_s2[true_label_idx_s2],
                'prediction_s2': self.class_labels_s2[pred_label_idx_s2],
                'confidence_s2': np.max(prediction_s2_raw) * 100,
                'full_prediction_s2': prediction_s2_raw # Сохраняем весь вектор вероятностей
            })
            
        return results
    
    def visualize_peak(self, window_data: np.ndarray, title: str = "Визуализация R-пика"):
        """
        Строит и отображает график для одного окна сигнала.
        
        :param window_data: Одномерный numpy-массив с данными окна.
        :param title: Заголовок для графика.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(window_data)
        plt.title(title)
        plt.xlabel("Отсчеты (samples)")
        plt.ylabel("Амплитуда")
        plt.grid(True)
        plt.show()

    def get_patient_list(self) -> list:
        """Возвращает отсортированный список уикальных ID пациентов."""
        patient_ids = {meta[0] for meta in self.metadata_labeled_s1}
        return sorted(list(patient_ids))

    def get_peaks_for_patient(self, patient_id: str) -> list:
        """
        Возвращает ПОЛНЫЙ, отсортированный по времени список R-пиков для пациента,
        размеченный по принадлежности к выборке (train/val/test).
        """
        if not patient_id:
            return []
        
        # Создаем множество тестовых сэмплов для быстрой проверки
        test_samples = {meta[1] for meta in self.data_stage1_test_only['metadata_test'] if str(meta[0]) == str(patient_id)} ## ДОБАВЛЕНА

        patient_peaks_text = []
        # Используем self.metadata_labeled_s1, так как в нем есть все пики
        for p_id, s_id, split_label in self.metadata_labeled_s1:
            if str(p_id) == str(patient_id):
                # Нашли пик нужного пациента. Формируем строку.
                # Формат: "Sample_ID [SPLIT_LABEL]"
                label_text = f"{s_id} [{split_label.upper()}]"

                # Добавляем специальную пометку для тестовых пиков
                if s_id in test_samples:
                    label_text += " 🧪"

                patient_peaks_text.append(label_text)
        return patient_peaks_text
    
    def _load_patient_summary(self) -> dict:
        """Загружает JSON-файл с детализированной статистикой по пикам пациентов."""
        summary_path = os.path.join(self.config['paths']['data_dir'], "patient_detailed_summary.json")
        if not os.path.exists(summary_path):
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл сводки {summary_path} не найден. 'Умный' список пациентов будет недоступен.")
            return {}
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"ОШИБКА при загрузке или парсинге JSON-сводки: {e}")
            return {}

    def _create_formatted_patient_list(self) -> list:
        """Создает список пациентов в формате 'ID 101 [N+(98.7%), N-(0.5%)]' для Dropdown."""
        if not self.patient_summary:
            return self.get_patient_list() # Возвращаем простой список, если сводка не загрузилась

        # Получаем "белый список" пациентов, у которых точно есть канал MLII,
        # из уже отфильтрованных метаданных.
        pids_with_mlii = {str(meta[0]) for meta in self.data_stage1_full['metadata']}

        formatted_list = []
        # Сортируем по ID пациента (ключи словаря)
        sorted_patient_ids = sorted(self.patient_summary.keys(), key=lambda x: int(x))

        for pid in sorted_patient_ids:
            # Проверяем, есть ли текущий пациент в нашем "белом списке".
            if pid not in pids_with_mlii:
                continue # Если нет - пропускаем и не добавляем в список.
            
            data = self.patient_summary[pid]
            distribution = data.get("distribution", {})
            
            # Сортируем классы по проценту (от большего к меньшему)
            sorted_classes = sorted(distribution.items(), key=lambda item: item[1]['total_percent'], reverse=True)
            
            stats = []
            for class_name, class_data in sorted_classes:
                percent = class_data['total_percent']

                # Применяем сокращения для Dropdown
                short_name = class_name
                if class_name == 'N+N':
                    short_name = 'N+'
                elif class_name == 'N (по Aux не N)':
                    short_name = 'N-'
                    
                stats.append(f"{short_name}({percent:.2f}%)")
                
            stats_str = ", ".join(stats)
            formatted_list.append(f"ID {pid} [{stats_str}]")
        
        return formatted_list
        
    def get_patient_stats_markdown(self, formatted_patient_str: str) -> str:
        """Возвращает markdown-отчет по выбранному пациенту для отображения в GUI."""
        if not formatted_patient_str or not self.patient_summary:
            return "Выберите пациента, чтобы увидеть подробную статистику."

        try:
            pid = formatted_patient_str.split(' ')[1]
        except IndexError:
            return "Ошибка: не удалось извлечь ID пациента."
        
        data = self.patient_summary.get(pid)
        if not data:
            return f"Статистика для пациента {pid} не найдена."

        total_peaks = data['total_peaks']
        distribution = data.get("distribution", {})
        
        report = [f"### Статистика по пациенту {pid}", f"**Всего R-пиков:** {total_peaks}\n"]

        # Сортируем для красивого вывода
        sorted_classes = sorted(distribution.items(), key=lambda item: item[1]['total_percent'], reverse=True)

        for class_name, class_data in sorted_classes:
            display_name = self.display_names.get(class_name, class_name)
            report.append(f"- **{display_name}:** {class_data['total_percent']:.2f}%")
            ### report.append(f"- **{class_name}:** {class_data['total_percent']}%")
            details = class_data.get('details', {})
            if len(details) > 1: # Показываем детали, только если в группе больше одного типа
                detail_parts = []
                for raw_type, percent in details.items():
                    detail_parts.append(f"{raw_type} ({percent}%)")
                report.append(f"  - `{' / '.join(detail_parts)}`")
        
        return "\n".join(report)
    
    def get_ecg_region_data(self, patient_id: str, center_percent: float) -> dict | None:
        """
        Загружает участок ЭКГ-сигнала и информацию о R-пиках для визуализации.

        :param patient_id: ID пациента.
        :param center_percent: Процент от начала записи, который будет центром визуализации (0-100).
        :return: Словарь с данными для отрисовки или None в случае ошибки.
        """
        # 1. Загрузка сырого сигнала
        signal_path = os.path.join(self.config['paths']['temp_dir'], f"{patient_id}.csv")
        if not os.path.exists(signal_path):
            print(f"ОШИБКА: Файл сигнала не найден: {signal_path}")
            return None
        
        try:
            df_signal = pd.read_csv(signal_path)
            # Очищаем имена колонок от кавычек и пробелов
            df_signal.columns = [col.strip().strip("'") for col in df_signal.columns]
            df_signal.rename(columns={"sample #": "Sample"}, inplace=True)

            target_channel = self.config['data']['target_channel_name']
            df_signal = df_signal[['Sample', target_channel]].rename(columns={target_channel: 'Signal'})

            ### # Загружаем только нужные колонки
            ### df_signal = pd.read_csv(signal_path, usecols=["'sample #'", f"'{self.config['data']['target_channel_name']}'"])
            ### df_signal.columns = ['Sample', 'Signal']
        except Exception as e:
            print(f"ОШИБКА при чтении CSV файла {signal_path}: {e}")
            return None

        total_samples = df_signal['Sample'].max()
        
        # 2. Определение границ буфера
        buffer_percent = config['data']['visualization']['buffer_percent']
        center_sample = int(total_samples * (center_percent / 100))

        start_sample = max(0, int(center_sample - (total_samples * buffer_percent / 100)))
        end_sample = min(total_samples, int(center_sample + (total_samples * buffer_percent / 100)))

        # Вырезаем кусок сигнала
        region_signal_df = df_signal[(df_signal['Sample'] >= start_sample) & (df_signal['Sample'] <= end_sample)]
        
        # 3. Фильтрация R-пиков
        # Сначала получаем все пики для пациента
        all_patient_peaks_info = [
            (p_id, s_id, split) for p_id, s_id, split in self.metadata_labeled_s1 if str(p_id) == str(patient_id)
        ]

        # Теперь фильтруем их по нашему диапазону
        region_peaks_info = [
            peak for peak in all_patient_peaks_info if start_sample <= peak[1] <= end_sample
        ]

        # Обогащаем пики реальным типом (N, V, A...) из первоисточника, который у нас уже загружен!
        # Это самый надежный способ.
        region_peaks = []
        for p_id, s_id, split in region_peaks_info:
            peak_type = '?' # Значение по умолчанию

            # Сначала ищем в индексе stage2 (для всех аномалий и N-)
            idx2 = self.metadata_index_s2.get((p_id, s_id))
            if idx2 is not None:
                y_ohe = self.data_stage2_full['y'][idx2]
                peak_type = self.class_labels_s2[np.argmax(y_ohe)]
            else:
                # Если не нашли, ищем в индексе stage1 (это должен быть N+N)
                idx1 = self.metadata_index_s1.get((p_id, s_id))
                if idx1 is not None:
                    # У stage1 бинарная метка, 0 - это 'Good (N+N)'
                    if int(self.data_stage1_full['y'][idx1]) == 0:
                        peak_type = 'N+N'

            region_peaks.append({
                'sample': s_id,
                'type': peak_type,
                'split': split
            })

        return {
            "signal_df": region_signal_df,
            "peaks": region_peaks,
            "total_samples": total_samples
        }