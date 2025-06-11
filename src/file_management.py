# file_management.py

import os
import shutil
import zipfile
import rarfile
import tarfile
import py7zr
from .config import config
from tqdm import tqdm  # <-- добавляем прогресс-бар

class FilePreparer:

    def __init__(self) -> None:
        self.input_dir = config['paths']['input_dir']
        self.temp_dir = config['paths']['temp_dir']
        os.makedirs(self.temp_dir, exist_ok=True)

    def pipeline(self) -> None:
        """
        Подготавливает файлы: распаковывает архивы, копирует файлы .txt, .csv во временную папку.
        """

        self._copy_files_to_temp()                              # Копируем .txt, .csv в self.temp_dir
        #self._extract_archives()                               # Если будут: Распаковываем все архивы в self.temp_dir

    def _copy_single_file_to_temp(self, single_file: str) -> str:
        base_name = os.path.basename(single_file)
        src = os.path.join(self.input_dir, base_name)
        dst = os.path.join(self.temp_dir, base_name)
        print(f"{src=}")
        print(f"{dst=}")
        shutil.copy(src, dst)
        return dst

    def _copy_files_to_temp(self):
        """Копирует в temp_dir все .txt, .csv из self.input_dir (включая подпапки)"""
        
        # Собираем все подходящие файлы
        all_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for f in files:
                if f.lower().endswith(('.csv', '.txt')):
                    all_files.append(os.path.join(root, f))

        # Прогресс-бар для всех файлов сразу
        for single_file_path in tqdm(all_files, desc="Копирование нужных файлов (.txt, .csv)", unit="файл"):
            self._copy_single_file_to_temp(single_file_path)

    def _extract_archives(self):
        """
        Распаковывает в temp_dir все .zip .tar .gz .7z .rar -архивы из self.input_dir.

        Использует пути, переданные при инициализации класса через self.input_dir и self.temp_dir.
        Не заходит в подпапки, работает только с файлами верхнего уровня.
        """
        # Поддерживаемые расширения
        archive_extensions = ('.zip', '.tar', '.tar.gz', '.tgz', '.targz', '.7z', '.rar')
        
        # Собираем все архивы рекурсивно
        all_archives = []
        for root, dirs, files in os.walk(self.input_dir):
            for f in files:
                if f.lower().endswith(archive_extensions):
                    all_archives.append(os.path.join(root, f))

        # Распаковываем с прогресс-баром
        for archive_path in tqdm(all_archives, desc="Распаковка архивов", unit="архив"):
            lower_f = archive_path.lower()
            try:
                # ZIP
                if lower_f.endswith('.zip'):
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(self.temp_dir)

                # TAR / GZ
                elif lower_f.endswith(('.tar', '.tar.gz', '.tgz', '.targz')):
                    mode = 'r'
                    if '.gz' in lower_f or '.tgz' in lower_f:
                        mode = 'r:gz'
                    with tarfile.open(archive_path, mode) as tf:
                        tf.extractall(self.temp_dir)

                # 7z
                elif lower_f.endswith('.7z'):
                    with py7zr.SevenZipFile(archive_path, mode='r') as z:
                        z.extractall(self.temp_dir)

                # RAR
                elif lower_f.endswith('.rar'):
                    with rarfile.RarFile(archive_path) as rf:
                        rf.extractall(self.temp_dir)

            except Exception as e:
                print(f"Ошибка при обработке архива {archive_path}: {e}")
