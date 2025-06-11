# main.py

#import src.config
from src.config import config

from src.file_management import FileManagement
from src.dataset_preparer import DatasetPreparation
from src.model_trainer import ModelTraining
from src.model_producer import ModelProduction

# Подготовка файлов
manager = FileManagement()
file_list = manager.pipeline()
	## выбор входящей директории, по умолчанию - папка INPUT в текущем каталоге (см. конфиг).
	## распаковка архивов
	## преоборазование файлов
	## удаление ненужных файлов
	## складирование енужных файлов в спец. директорию

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
producer = ModelProduction()
producer.pipeline()