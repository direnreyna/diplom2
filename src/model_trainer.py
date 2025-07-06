# model_trainer.py

import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from self_attention_block import SelfAttentionBlock
from typing import Tuple, List
from config import config

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, Activation, Conv1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTraining:
    def __init__(self, stage, prefix) -> None:
        self.stage = stage
        self.multi_stages = config['stages']['multi']
        self.prefix = prefix
        self.model = None
        self.history = None
        self.class_labels = config['class_labels']
        
        # Имя файла для ПОЛНОЙ модели (архитектура, веса, оптимизатор)
        self.best_model_file = os.path.join(config['paths']['model_dir'], f"{self.prefix}_{self.stage}_{config['paths']['best_model']}")   ## Сохраненная модель для стадии
        
        # Имя файла ТОЛЬКО для ВЕСОВ
        self.best_model_file_weights = os.path.join(config['paths']['model_dir'], f"{self.prefix}_{self.stage}_{config['paths']['best_model_weights']}") ## Сохраненные веса

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        num_classes = len(np.unique(self.y_train))      # 2: Good (0), Alert (1)
        os.makedirs(config['paths']['model_dir'], exist_ok=True)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Загружает уже сохраненный ранее датасет"""

        dataset_path = config['paths']['data_dir']
        dataset_name = config['data']['dataset_name']
        file = f"{self.prefix}_{self.stage}_{dataset_name}"

        file_dataset = os.path.join(dataset_path, file)
        if not os.path.exists(file_dataset):
            raise FileNotFoundError(f"Не обнаружен датасет: {file_dataset}")
        data = np.load(file_dataset)
        return (data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'])
    
    def pipeline(self, mode: str = 'full') -> None:
        """
        Полный пайплайн: подготовка модели → обучение → оценка → сохранение лучшей модели

        :param управляет логикой пайплайна mode:
            - 'full' (подготовка модели → обучение → оценка → сохранение лучшей модели)
            - 'eval' (только инференс)
        :param force_rebuild: True, если изменена архитектура и обучение нужно начать с нуля, игнорируя сохраненные веса.
        """

        self._create_model()
        if mode == 'full':
            self._train_model()         ## Обучение / Дообучение модели
        self._evaluate_on_val()
        self.evaluate_model()

    def _categorical_focal_loss(self, gamma: float = 2.0, alpha: List[float] | None = None):
        """
        Focal Loss для многоклассовой классификации
        :param gamma: фокусировка на сложных примерах
        :param alpha: список весов для каждого класса
        """
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)

        def loss(y_true, y_pred):
            # Сглаживаем предсказания
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

            # Вероятность истинного класса
            pt = tf.reduce_sum(y_true * y_pred, axis=-1)

            # Веса альфа для истинного класса
            weight = tf.reduce_sum(y_true * alpha_tensor, axis=-1)

            # Формула Focal Loss
            fl = -weight * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)

            return fl  # минимизируем эту величину
        return loss

    def f1_score(self, y_true, y_pred):
        is_multiclass = y_true.shape[-1] is not None and y_true.shape[-1] > 1

        # Корректная логика для мультиклассовой классификации
        if is_multiclass:
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            y_true_labels = tf.argmax(y_true, axis=-1)
            
            cm = tf.math.confusion_matrix(y_true_labels, y_pred_labels, num_classes=tf.shape(y_true)[-1])
            
            tp = tf.linalg.diag_part(cm)
            fp = tf.reduce_sum(cm, axis=0) - tp
            fn = tf.reduce_sum(cm, axis=1) - tp

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            
            return tf.reduce_mean(f1)

        # Корректная логика для бинарной классификации
        else:
            y_pred_rounded = tf.round(tf.clip_by_value(y_pred, 0, 1))

            tp = tf.reduce_sum(y_true * y_pred_rounded)
            fp = tf.reduce_sum((1 - y_true) * y_pred_rounded)
            fn = tf.reduce_sum(y_true) - tp

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)

            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            return f1

    def _create_model(self):
        """
        Создает или загружает модель, реализуя логику дообучения.
        """
        #################################################################################
        # Сценарий 1: ДООБУЧЕНИЕ С ТЕМИ ЖЕ ПАРАМЕТРАМИ или ОЦЕНКА
        #################################################################################
        # Если существует файл полной модели, мы всегда предпочитаем его,
        # так как он содержит состояние оптимизатора.
        if os.path.exists(self.best_model_file):
            print(f"Обнаружен файл полной модели. Загружаем из: {self.best_model_file}")
            self.model = load_model(
                self.best_model_file, 
                custom_objects={'SelfAttentionBlock': SelfAttentionBlock, 'f1_score': self.f1_score}
            )
            print("Полная модель успешно загружена, включая состояние оптимизатора.")
            self.model.summary()
            return
        
        # Если мы здесь, значит файла полной модели нет.
        # Это сценарий ПЕРВОГО ЗАПУСКА или ИЗМЕНЕНИЯ АРХИТЕКТУРЫ.
        print("Файл полной модели не найден. Создаем новую архитектуру...")

        #################################################################################
        # Сценарий 2: ПЕРВЫЙ ЗАПУСК или ИЗМЕНЕНИЕ АРХИТЕКТУРЫ.
        #################################################################################

        #################################################################################
        if self.stage in self.multi_stages:
            class_counts = np.sum(self.y_train, axis=0)
            total = np.sum(class_counts)
            alpha_weights = np.sqrt(total / class_counts)               # или любая другая стратегия
            alpha_weights /= np.sum(alpha_weights)                      # нормализуем
            stage_loss=self._categorical_focal_loss(gamma=2.0, alpha=alpha_weights)
            last_layer = Dense(self.y_train.shape[1], activation='softmax')
            stage_metrics=['accuracy', self.f1_score]
            print(f">>> Число классов: {self.y_train.shape[1]}")

        else:
            stage_loss='binary_crossentropy'
            last_layer = Dense(1, activation='sigmoid')
            stage_metrics=['accuracy']
        #################################################################################

        if len(self.X_train.shape) < 3:
            self.X_train = np.expand_dims(self.X_train, axis=2)
        input_shape = self.X_train.shape[1:]            # форма экземпляра: (window_size, channels)

        print(f"Создаем новую модель.")
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv1D(64, (3), activation='relu', input_shape=input_shape))
        model.add(SelfAttentionBlock(use_projection=True))
        model.add(Conv1D(128, (3), activation='relu'))
        model.add(SelfAttentionBlock(use_projection=True))
        
        ### model.add(Conv1D(64, (3), input_shape=input_shape))
        ### model.add(BatchNormalization())
        ### model.add(Activation('relu'))
        ### model.add(SelfAttentionBlock(use_projection=True))
        ### #model.add(Dropout(0.3))
        ### model.add(Conv1D(128, (3)))
        ### model.add(BatchNormalization())
        ### model.add(Activation('relu'))
        ### model.add(SelfAttentionBlock(use_projection=True))
        ### #model.add(Dropout(0.3))

        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.4))

        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.4))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(last_layer)

        model.compile(
            optimizer=Adam(learning_rate=config['params']['learning_rate']),
            loss=stage_loss,
            metrics=stage_metrics
        )
        model.summary()

        #################################################################################
        # Сценарий 3: Дообучение модели без ИЗМЕНЕНИЯ АРХИТЕКТУРЫ (изменение оптимизатора)
        #################################################################################
        # Загрузка весов сохраненной дообученной модели
        if os.path.exists(self.best_model_file_weights):
            print(f"Загружаем ВЕСА в новую архитектуру из: {self.best_model_file_weights}.")
            try:
                model.load_weights(self.best_model_file_weights)
                print("Веса успешно загружены.")
            except ValueError as e:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить веса. Ошибка: {e}. Обучение начнется с нуля.")
        self.model = model

    def _train_model(self):
        """
        Обучает модель на тренировочных данных с валидацией
        """
        if self.model is None:
            print("ОШИБКА: Модель не была создана или загружена. Прерывание обучения.")
            return

        epochs=config['params']['epochs']
        batch_size=config['params']['batch_size']

        if self.stage in self.multi_stages:
            monitor = 'val_f1_score'
        else:
            monitor = 'val_accuracy'

        ## сохраняем всю модель
        best_full_model = ModelCheckpoint( 
            self.best_model_file,               
            monitor=monitor,               
            save_best_only=True,                ## сохраняем лучшую модель...
            save_weights_only=False,            ## ... при этом НЕ только веса
            mode='max',                    
            verbose=1)                     
        
        ## сохраняем только веса
        best_weights_only = ModelCheckpoint(    
            self.best_model_file_weights,       
            monitor=monitor,
            save_best_only=True,                ## сохраняем лучшую модель...
            save_weights_only=True,             ## ... при этом только веса
            mode='max',
            verbose=1)

        early_stop = EarlyStopping(
            monitor=monitor,
            patience=config['params']['patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[best_full_model, best_weights_only, early_stop],
            verbose=1
        )
        self.history = history
        self._plot_training_history()

        # Выведем пару случайных окон
        for i in range(5):
            z = np.random.randint(1000)
            plt.plot(self.X_train[i+z])
            plt.title(f"Label: {self.y_train[i+z]}")
            plt.show()        
        
        print(f"[{self.prefix}] Обучение модели завершено")

    def _plot_training_history(self):
        if self.history is None:
            raise ValueError("Нет данных об обучении. Обучите модель сначала.")

        print(f"[{self.prefix}] Визуализируем метрики обучения")

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        if self.stage in self.multi_stages:

            f1_score = self.history.history['f1_score']
            val_f1_score = self.history.history['val_f1_score']

            plt.figure(figsize=(18, 5))

            # Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(epochs, acc, 'b', label='Train accuracy')
            plt.plot(epochs, val_acc, 'r', label='Valid accuracy')
            plt.title(f'{self.prefix} - Accuracy')
            plt.legend()

            # f1-score
            plt.subplot(1, 3, 2)
            plt.plot(epochs, f1_score, 'b', label='Train f1-score')
            plt.plot(epochs, val_f1_score, 'r', label='Valid f1-score')
            plt.title(f'{self.prefix} - f1-score')
            plt.legend()

            # Loss
            plt.subplot(1, 3, 3)
            plt.plot(epochs, loss, 'b', label='Train loss')
            plt.plot(epochs, val_loss, 'r', label='Valid loss')
            plt.title(f'{self.prefix} - Loss')
            plt.legend()

        else:
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(epochs, acc, 'b', label='Train accuracy')
            plt.plot(epochs, val_acc, 'r', label='Valid accuracy')
            plt.title(f'{self.prefix} - Accuracy')
            plt.legend()

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(epochs, loss, 'b', label='Train loss')
            plt.plot(epochs, val_loss, 'r', label='Valid loss')
            plt.title(f'{self.prefix} - Loss')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix", save_path=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _evaluate_on_val(self):
        """
        Оценивает качество модели на валидационной выборке
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала обучите модель.")

        print(f"[{self.prefix}] Оценка на валидационной выборке")
        y_pred = self.model.predict(self.X_val)

        # многоклассовая
        if self.stage in self.multi_stages:
            y_true = np.argmax(self.y_val, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
        # бинарная 
        else:
            y_true = self.y_val.astype(int).flatten()
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()

        self._print_evaluation(y_true, y_pred_classes, dataset_name="валидационной")
        
    def evaluate_model(self):
        """
        Оценивает качество модели на тестовой выборке
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала обучите модель.")

        print(f"[{self.prefix}] Оценка на тестовой выборке")
        y_pred = self.model.predict(self.X_test)

        # многоклассовая
        if self.stage in self.multi_stages:
            y_true = np.argmax(self.y_test, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
        # бинарная 
        else:
            y_true = self.y_test.astype(int).flatten()
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()

        self._print_evaluation(y_true, y_pred_classes, dataset_name="тестовой")

    def _print_evaluation(self, y_true, y_pred_classes, dataset_name=""):
        """
        Выводит метрики качества: accuracy, precision, recall, f1-score
        """
        acc = accuracy_score(y_true, y_pred_classes)
        print(f"\nAccuracy на {dataset_name} выборке: {acc:.4f}\n")

        report = classification_report(
            y_true, 
            y_pred_classes,
            target_names=self.class_labels[self.stage],
            digits=4
        )
        print(report)

        # Матрица ошибок (опционально)
        class_names = self.class_labels[self.stage]  # список имён классов
        cm = confusion_matrix(y_true, y_pred_classes)
        print("Confusion matrix:\n", cm)
        self.plot_confusion_matrix(cm, class_names=class_names)

    def evaluate_on_test(self):
        if self.model is None:
            raise ValueError("Модель не создана. Сначала обучите модель.")

        print(f"[{self.prefix}] Оценка на тестовой выборке")
        y_pred = self.model.predict(self.X_test)

        # многоклассовая
        if self.stage in self.multi_stages:
            y_true = np.argmax(self.y_test, axis=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
        # бинарная
        else:
            y_true = self.y_test.astype(int).flatten()
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()

        report = classification_report(y_true, y_pred_classes, digits=4)
        print(report)

        class_names = self.class_labels[self.stage]  # список имён классов
        cm = confusion_matrix(y_true, y_pred_classes)
        print("Confusion matrix:\n", cm)
        self.plot_confusion_matrix(cm, class_names=class_names)
        
        return {
            'report': report,
            'confusion_matrix': cm
        }
    

