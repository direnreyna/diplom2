# model_trainer.py

import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from config import config
from tqdm import tqdm
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTraining:
    def __init__(self, prefix) -> None:
        self.prefix = prefix
        self.model = None
        self.history = None
        self.best_model_file = os.path.join(config['paths']['model_dir'], f"{self.prefix}_{config['paths']['best_model']}")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        num_classes = len(np.unique(self.y_train))      # 2: Good (0), Alert (1)
        os.makedirs(config['paths']['model_dir'], exist_ok=True)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Загружает уже сохраненный ранее датасет"""
        
        dataset_name = config['data']['dataset_name']
        file = self.prefix + "_" + dataset_name
        dataset_path = config['paths']['temp_dir']
        file_dataset = os.path.join(dataset_path, file)
        if not os.path.exists(file_dataset):
            raise FileNotFoundError(f"Не обнаружен датасет: {file_dataset}")
        data = np.load(file_dataset)
        return (data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'])
    
    def pipeline(self) -> None:
        """Полный пайплайн: подготовка модели → обучение → оценка → сохранение"""
        self._create_model()
        self._train_model()
        self._evaluate_on_val()
        self.evaluate_model()

    def _create_model(self):
        """
        Создает и компилирует модель
        """
        input_shape = self.X_train.shape[1:]            # форма экземпляра: (window_size, channels)

        # Загрузка сохраненной дообученной модели
        if os.path.exists(self.best_model_file):
            print(f"Загружаем модель с диска: {self.best_model_file}.")
            self.model = load_model(self.best_model_file)
            return

        # Загрузка предобученной модели с сайта
        print(f"Создаем новую модель.")
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.4))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        self.model = model

    def _train_model(self):
        """
        Обучает модель на тренировочных данных с валидацией
        """
        epochs=config['params']['epochs']
        batch_size=config['params']['batch_size']
        
        best_model = ModelCheckpoint(
            self.best_model_file,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config['params']['patience'],
            restore_best_weights=True,
            verbose=1)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[best_model, early_stop],
            verbose=1
        )
        self.history = history
        self._plot_training_history()
        
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

    def _evaluate_on_val(self):
        """
        Оценивает качество модели на валидационной выборке
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала обучите модель.")

        print(f"[{self.prefix}] Оценка на валидационной выборке")
        y_pred = self.model.predict(self.X_val)
        y_pred_classes = (y_pred > 0.5).astype(int)  # бинарная классификация
        y_true = self.y_val

        self._print_evaluation(y_true, y_pred_classes, dataset_name="валидационной")
        
    def evaluate_model(self):
        """
        Оценивает качество модели на тестовой выборке
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала обучите модель.")

        print(f"[{self.prefix}] Оценка на тестовой выборке")
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)
        y_true = self.y_test

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
            target_names=["Good (0)", "Alert (1)"],
            digits=4
        )
        print(report)

        # Матрица ошибок (опционально)
        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"Матрица ошибок ({dataset_name}):\n", cm)
