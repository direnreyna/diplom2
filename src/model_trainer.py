# src/model_trainer.py

import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
# import mlflow.keras
# import mlflow.tensorflow

from self_attention_block import SelfAttentionBlock
from typing import Tuple, cast
from config import config

from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Model, losses, saving, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, Activation, Conv1D, MaxPooling1D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

@tf.keras.saving.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss): 
    """Классовая реализация Focal Loss. Гарантирует корректную сериализацию."""    

    def __init__(self, gamma=2.0, alpha=[1.0, 1.0, 1.0, 1.0], name="categorical_focal_loss", **kwargs): 
        super().__init__(name=name, **kwargs) 
        self.gamma = gamma 
        self.alpha = alpha 

    def call(self, y_true, y_pred): 
        alpha_tensor = tf.constant(self.alpha, dtype=tf.float32) 

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7) 
        pt = tf.reduce_sum(y_true * y_pred, axis=-1) 
        weight = tf.reduce_sum(y_true * alpha_tensor, axis=-1) 
        fl = -weight * tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt) 
        return fl 

    def get_config(self): 
        # Этот метод позволяет Keras сохранять параметры loss'a 
        config = super().get_config() 
        config.update({ 
            "gamma": self.gamma, 
            "alpha": self.alpha 
        }) 
        return config 

@tf.keras.saving.register_keras_serializable()
def f1_score(y_true, y_pred):
    is_multiclass = y_true.shape[-1] is not None and y_true.shape[-1] > 1

    # Корректная логика для мультиклассовой классификации
    if is_multiclass:
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        y_true_labels = tf.argmax(y_true, axis=-1)
        
        cm = tf.math.confusion_matrix(y_true_labels, y_pred_labels, num_classes=y_true.shape[-1])
        
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp

        tp = tf.cast(tp, dtype=tf.float32)
        fp = tf.cast(fp, dtype=tf.float32)
        fn = tf.cast(fn, dtype=tf.float32)

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

@tf.keras.saving.register_keras_serializable()
class MacroF1Score(tf.keras.metrics.Metric):
    """
    Keras Metric для вычисления Macro F1-Score.
    Накапливает Confusion Matrix за эпоху и вычисляет F1-меру один раз в конце.
    """
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Накапливаем компоненты Confusion Matrix
        # Важно: инициализируем как tf.Variable, чтобы они были частью состояния метрики
        self.tp = self.add_weight(name='tp', shape=(self.num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(self.num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(self.num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Преобразуем OHE в sparse labels
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        
        # Вычисляем Confusion Matrix для текущего батча
        cm_batch = tf.math.confusion_matrix(y_true_labels, y_pred_labels, num_classes=self.num_classes)
        
        # Накапливаем TP, FP, FN
        tp_batch = tf.linalg.diag_part(cm_batch)
        fp_batch = tf.reduce_sum(cm_batch, axis=0) - tp_batch
        fn_batch = tf.reduce_sum(cm_batch, axis=1) - tp_batch

        self.tp.assign_add(tf.cast(tp_batch, dtype=tf.float32))
        self.fp.assign_add(tf.cast(fp_batch, dtype=tf.float32))
        self.fn.assign_add(tf.cast(fn_batch, dtype=tf.float32))

    def result(self):
        # Вычисляем Precision и Recall. Используем tf.math.divide_no_nan для стабильности.
        precision = tf.math.divide_no_nan(self.tp, (self.tp + self.fp))
        recall = tf.math.divide_no_nan(self.tp, (self.tp + self.fn))
        
        # Вычисляем F1-меру для каждого класса
        f1_per_class = tf.math.divide_no_nan(2 * precision * recall, (precision + recall))
        
        # Возвращаем Macro F1 (среднее по всем классам)
        return tf.reduce_mean(f1_per_class)

    def reset_states(self):
        # Сбрасываем накопленные состояния в начале каждой эпохи
        K.set_value(self.tp, tf.zeros(shape=(self.num_classes,)))
        K.set_value(self.fp, tf.zeros(shape=(self.num_classes,)))
        K.set_value(self.fn, tf.zeros(shape=(self.num_classes,)))

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

class ModelTraining:
    def __init__(self, stage, prefix, load_from_mlflow=False, mlflow_run_id=None) -> None:
        self.stage = stage
        self.multi_stages = config['stages']['multi']
        self.prefix = prefix
        self.load_from_mlflow = load_from_mlflow
        self.mlflow_run_id = mlflow_run_id
        self.model = None
        self.history = None
        self.class_labels = config['class_labels']
        
        # Имя файла для ПОЛНОЙ модели (архитектура, веса, оптимизатор)
        self.best_model_file = os.path.join(config['paths']['model_dir'], f"{self.prefix}_{self.stage}_{config['paths']['best_model']}")   ## Сохраненная модель для стадии
        
        # Имя файла ТОЛЬКО для ВЕСОВ
        self.best_model_file_weights = os.path.join(config['paths']['model_dir'], f"{self.prefix}_{self.stage}_{config['paths']['best_model_weights']}") ## Сохраненные веса

        ### self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.metadata_train, self.metadata_val, self.metadata_test = self.load_dataset()

        os.makedirs(config['paths']['model_dir'], exist_ok=True)

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Загружает уже сохраненный ранее датасет"""

        dataset_path = config['paths']['data_dir']
        dataset_name = config['data']['dataset_name']
        file = f"{self.prefix}_{self.stage}_{dataset_name}"

        file_dataset = os.path.join(dataset_path, file)
        if not os.path.exists(file_dataset):
            raise FileNotFoundError(f"Не обнаружен датасет: {file_dataset}")
        data = np.load(file_dataset, allow_pickle=True)
        return (data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['metadata_train'], data['metadata_val'], data['metadata_test'])
    
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

    def _create_model(self):
        """
        Создает или загружает модель, реализуя логику дообучения:

        Сценарий 0: ПРИОРИТЕТНАЯ ЗАГРУЗКА ИЗ MLFLOW - Если включен флаг load_from_mlflow
        Сценарий 1: ДООБУЧЕНИЕ С ТЕМИ ЖЕ ПАРАМЕТРАМИ или ОЦЕНКА - Если существует файл лучшей модели
        Сценарий 2: ПЕРВЫЙ ЗАПУСК или ИЗМЕНЕНИЕ АРХИТЕКТУРЫ - Если файла лучшей модели нет
        Сценарий 3: Дообучение модели без ИЗМЕНЕНИЯ АРХИТЕКТУРЫ (изменение оптимизатора) - Загрузка весов сохраненной дообученной модели
        """

        #################################################################################
        # Сценарий 0: Приоритетная загрузка из MLflow
        #################################################################################
        if self.load_from_mlflow and self.mlflow_run_id:
            print(f"Приоритетная загрузка модели из MLflow run_id: {self.mlflow_run_id}")
            try:
                # Формируем URI для загрузки артефакта модели, который был
                model_uri = f"runs:/{self.mlflow_run_id}/model"

                # Создаем словарь с кастомными объектами для загрузки
                custom_objects_to_load = {
                    'SelfAttentionBlock': SelfAttentionBlock,
                    'CategoricalFocalLoss': CategoricalFocalLoss,
                    'MacroF1Score': MacroF1Score                            ## Используем класс метрики вместо функции
                    #'f1_score': f1_score
                }
                loaded_model = mlflow.keras.load_model(model_uri, custom_objects=custom_objects_to_load)
                # Явно указываем анализатору, что это модель Keras
                self.model = cast(Model, loaded_model)
                if self.model:
                    print("Модель из MLflow успешно загружена.")
                    self.model.summary()
                    return
                else:
                    print("ОШИБКА: MLflow вернул None вместо модели без вызова исключения.")

            except Exception as e:
                print(f"ОШИБКА: Не удалось загрузить модель из MLflow. Ошибка: {e}")
                print("Продолжаем выполнение по стандартной логике (поиск локальных файлов).")

        #################################################################################
        # Сценарий 1: ДООБУЧЕНИЕ С ТЕМИ ЖЕ ПАРАМЕТРАМИ или ОЦЕНКА
        #################################################################################
        # Если существует файл полной модели, мы всегда предпочитаем его,
        # так как он содержит состояние оптимизатора.
        if os.path.exists(self.best_model_file):
            print(f"Обнаружен файл полной модели. Загружаем из: {self.best_model_file}")

            ### custom_objects_to_load = {
            ###     'SelfAttentionBlock': SelfAttentionBlock,
            ###     'CategoricalFocalLoss': CategoricalFocalLoss,
            ###     'f1_score': f1_score
            ### }
            ### 
            ### self.model = load_model( ## Изменена
            ###     self.best_model_file, 
            ###     custom_objects=custom_objects_to_load ## Изменена
            ### )

            self.model = load_model(
                self.best_model_file, 
                # custom_objects={'SelfAttentionBlock': SelfAttentionBlock, 'f1_score': self.f1_score, 'loss':self._categorical_focal_loss}
                custom_objects={
                    'SelfAttentionBlock': SelfAttentionBlock,
                    'MacroF1Score': MacroF1Score                            ## Используем класс метрики вместо функции
                    #'f1_score': self.f1_score,
                    #'loss': self.categorical_focal_loss
                    ###'categorical_focal_loss': self.categorical_focal_loss
                } 
            )
            print("Полная модель успешно загружена, включая состояние оптимизатора.")
            self.model.summary()
            return
        
        #################################################################################
        # Сценарий 2: ПЕРВЫЙ ЗАПУСК или ИЗМЕНЕНИЕ АРХИТЕКТУРЫ.
        # Если мы здесь, значит файла полной модели нет.
        #################################################################################
        print("Файл полной модели не найден. Создаем новую архитектуру...")

        #################################################################################
        if self.stage in self.multi_stages:
            class_counts = np.sum(self.y_train, axis=0)
            total = np.sum(class_counts)

            # Попытка сильнее регулировать штрафы при дисбалансе классов.
            #alpha_weights = total / class_counts                     
            alpha_weights = np.sqrt(total / class_counts)             
            alpha_weights /= np.sum(alpha_weights)                                             # нормализуем
            #stage_loss=self.categorical_focal_loss(gamma=2.0, alpha=alpha_weights)
            ##stage_loss=categorical_focal_loss(gamma=2.0, alpha=alpha_weights)
            stage_loss=CategoricalFocalLoss(gamma=2.0, alpha=alpha_weights.tolist())

            last_layer = Dense(self.y_train.shape[1], activation='softmax')
            #stage_metrics=['accuracy', self.f1_score]
            ### stage_metrics=['accuracy', f1_score]
            stage_metrics=['accuracy', MacroF1Score(num_classes=self.y_train.shape[1])]       ## Используем класс метрики вместо функции

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
        #################################################################################
        # Переход на функциональное написание модели для введения MHA блоков и резидуал коннекшн
        attention_type = config['params']['attention_type']
        print(f"Создаем новую модель с типом внимания: {attention_type}")

        input_layer_functional = Input(shape=input_shape)
        
        # Стартовая ветка для MULTI-HEAD ATTENTION
        if attention_type == 'multi_head_attention':
            
            # Трансформерный блок 1
            x = Conv1D(32, (3), activation='relu', kernel_regularizer=l2(0.0001))(input_layer_functional)
            
            attention_input = x
            norm_input = LayerNormalization()(attention_input)
            attention_output = MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)(norm_input, norm_input)
            x = attention_input + attention_output

            ffn_input = x
            norm_input = LayerNormalization()(ffn_input)
            ffn_output = Dense(32*4, activation="relu", kernel_regularizer=l2(0.001))(norm_input)
            ffn_output = Dense(32, kernel_regularizer=l2(0.001))(ffn_output)
            x = ffn_input + ffn_output

            # Трансформерный блок 2
            x = Conv1D(64, (3), activation='relu', kernel_regularizer=l2(0.0001))(x)

            attention_input = x
            norm_input = LayerNormalization()(attention_input)
            attention_output = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(norm_input, norm_input)
            x = attention_input + attention_output

            ffn_input = x
            norm_input = LayerNormalization()(ffn_input)
            ffn_output = Dense(64*4, activation="relu", kernel_regularizer=l2(0.001))(norm_input)
            ffn_output = Dense(64, kernel_regularizer=l2(0.001))(ffn_output)
            x = ffn_input + ffn_output

            ### # Трансформерный блок 3
            ### x = Conv1D(256, (3), activation='relu', kernel_regularizer=l2(0.001))(x)
            ### attention_input_3 = x
            ### attention_output_3 = MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1)(attention_input_3, attention_input_3)
            ### x = LayerNormalization()(attention_input_3 + attention_output_3)

        # Стартовая ветка для SELF-ATTENTION
        else:
            x = Conv1D(128, (3), activation='relu', kernel_regularizer=l2(0.0001))(input_layer_functional)
            x = SelfAttentionBlock(use_projection=True)(x)
            x = Conv1D(256, (3), activation='relu', kernel_regularizer=l2(0.0001))(x)
            x = SelfAttentionBlock(use_projection=True)(x)

        # Общая часть модели для обеих веток
        ### x = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.00001))(x)
        ### x = Dropout(0.2)(x)

        x = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.00001))(x)
        x = Dropout(0.4)(x)
        x = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.00001))(x)
        x = Dropout(0.4)(x)
        x = Dense(32, kernel_regularizer=l2(0.00001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        output_layer = last_layer(x)
        
        ## Собираем модель, указывая входы и выходы
        model = Model(inputs=input_layer_functional, outputs=output_layer)

        model.compile(
            optimizer=Adam(learning_rate=config['params']['learning_rate']),
            loss=stage_loss,
            metrics=stage_metrics
        )
            
        #################################################################################
        ### Вариант Sequential для архива
        ###    model = Sequential()
        ###    model.add(InputLayer(input_shape=input_shape))
        ###    model.add(Conv1D(128, (3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.0001)))
        ###    model.add(SelfAttentionBlock(use_projection=True))
        ###    model.add(Conv1D(256, (3), activation='relu', kernel_regularizer=l2(0.0001)))
        ###    model.add(SelfAttentionBlock(use_projection=True))
        ###    model.add(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.00001)))
        ###    model.add(Dropout(0.4))
        ###    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.00001)))
        ###    model.add(Dropout(0.4))
        ###    model.add(Dense(32, kernel_regularizer=l2(0.00001)))
        ###    model.add(BatchNormalization())
        ###    model.add(Activation('relu'))
        ###    model.add(last_layer)
        ###
        ###    model.compile(
        ###        optimizer=Adam(learning_rate=config['params']['learning_rate']),
        ###        loss=stage_loss,
        ###        metrics=stage_metrics
        ###    )
        model.summary()
        #################################################################################

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
            #monitor = 'val_f1_score'
            monitor = 'val_macro_f1_score'      ## Используем класс метрики вместо функции
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
            patience=config['params']['patience_early_stop'],
            restore_best_weights=True,
            mode='max',
            verbose=1)

        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=config['params']['factor_reduce_lr'],
            patience=config['params']['patience_reduce_lr'],
            min_lr=config['params']['min_learning_rate'],
            verbose=1)

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[best_full_model, best_weights_only, early_stop, reduce_lr],
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

        if self.stage in self.multi_stages:

            macro_f1_score = self.history.history['macro_f1_score']
            val_macro_f1_score = self.history.history['val_macro_f1_score']

            plt.figure(figsize=(18, 5))

            # Accuracy
            plt.subplot(1, 3, 1)
            plt.plot(epochs, acc, 'b', label='Train accuracy')
            plt.plot(epochs, val_acc, 'r', label='Valid accuracy')
            plt.title(f'{self.prefix} - Accuracy')
            plt.legend()

            # f1-score
            plt.subplot(1, 3, 2)
            plt.plot(epochs, macro_f1_score, 'b', label='Train Macro f1-score')
            plt.plot(epochs, val_macro_f1_score, 'r', label='Valid Macro f1-score')
            plt.title(f'{self.prefix} - Macro f1-score')
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
        report_text = classification_report(
            y_true, 
            y_pred_classes,
            target_names=self.class_labels[self.stage],
            digits=4
        )

        # Логируем финальные метрики для MLflow
        report_dict_raw = classification_report(
            y_true, 
            y_pred_classes,
            target_names=self.class_labels[self.stage],
            digits=4,
            output_dict=True
        )
        report_dict = cast(dict, report_dict_raw)

        # Логируем метрики для всей выборки (accuracy, macro avg)
        mlflow.log_metric(f"{dataset_name}_accuracy", report_dict['accuracy'])
        mlflow.log_metric(f"{dataset_name}_macro_f1_score", report_dict['macro avg']['f1-score'])
        mlflow.log_metric(f"{dataset_name}_weighted_f1_score", report_dict['weighted avg']['f1-score'])
        mlflow.log_text(str(report_text), f"report_{dataset_name}.txt")

        print(f"\nAccuracy на {dataset_name} выборке: {report_dict['accuracy']:.4f}\n")
        print(report_text)

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
