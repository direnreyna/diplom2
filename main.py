# main.py

import os
import sys
# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать модули
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

#import src.config
from config import config

from file_management import FileManagement
from dataset_preparer import DatasetPreparation
from model_trainer import ModelTraining
from model_producer import Production

# Подготовка файлов
manager = FileManagement()
file_list = manager.pipeline()
	## выбор входящей директории, по умолчанию - папка INPUT в текущем каталоге (см. конфиг).
	## распаковка архивов
	## преоборазование файлов
	## удаление ненужных файлов
	## складирование нужных файлов в спец. директорию.

# Подготовка датасета
preparer = DatasetPreparation()
(x_train, y_train, x_val, y_val, x_test, y_test) = preparer.pipeline(file_list)
	## загрузка датасета из файлов по списку file_list
	## сбор DF
	## получение дополнительных параметров:
	## 	типа скорости и ускорения изменения основных характеристик,
	## 	получение карт R-пиков и т.д.
	## label-классификация, перевод в ohe
	## нормирование/стандартизирование
	## разделение X, y на выборки x_train, y_train, x_val, y_val, x_test, y_test

# Обучение модели
trainer = ModelTraining()
trainer.pipeline(x_train, y_train, x_val, y_val, x_test, y_test)

# Инференс модели
producer = Production()
producer.pipeline()