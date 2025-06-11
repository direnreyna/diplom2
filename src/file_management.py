# file_management.py

import os
import shutil
import zipfile
import rarfile
import tarfile
import py7zr
from .config import INPUT_DIR, TEMP_DIR
from typing import Union
from docx import Document
from tqdm import tqdm  # <-- добавляем прогресс-бар

class FilePreparer:

    def __init__(self) -> None:
        os.makedirs(TEMP_DIR, exist_ok=True)

    def prepare_files(self) -> list[str]:
        """
        Подготавливает файлы: распаковывает архивы, копирует файлы .docx во временную папку.
        
        Выход:
            list[str]: Список путей ко всем файлам 
        """

        self._copy_files_to_temp()                              # Копируем .txt, .docx в TEMP_DIR
        self._extract_archives()                                # Распаковываем все архивы в TEMP_DIR
        self._convert_all_docx_in_temp()                        # Конвертируем все .docx в .txt
        self._remove_non_txt()                                  # Удаляем все файлы не-.txt
        return self._get_text_paths()                           # Возвращаем список .txt из TEMP_DIR

    def _copy_single_file_to_temp(self, single_file: str) -> str:
        base_name = os.path.basename(single_file)
        src = os.path.join(INPUT_DIR, base_name)
        dst = os.path.join(TEMP_DIR, base_name)
        print(f"{src=}")
        print(f"{dst=}")
        shutil.copy(src, dst)
        return dst

    def _copy_files_to_temp(self):
        """Копирует в temp_dir все .txt, .docx из INPUT_DIR"""
        need_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.docx', '.txt'))]
        # Оборачиваем итерацию в tqdm для прогресс-бара
        for single_file in tqdm(need_files, desc="Копирование нужных файлов (.txt, .docx)", unit="файл"):
            if single_file.lower().endswith(('.docx', '.txt')):
                _ = self._copy_single_file_to_temp(single_file)

    def _extract_archives(self):
        """
        Распаковывает в temp_dir все .zip .tar .gz .7z .rar -архивы из INPUT_DIR.

        Использует пути, переданные при инициализации класса через INPUT_DIR и TEMP_DIR.
        Не заходит в подпапки, работает только с файлами верхнего уровня.
        """
        for f in os.listdir(INPUT_DIR):
            path = os.path.join(INPUT_DIR, f)
            lower_f = f.lower()

            # ZIP-архивы
            if lower_f.endswith('.zip'):
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(TEMP_DIR)
                except zipfile.BadZipFile:
                    print(f"Ошибка чтения ZIP-архива: {path}")

            # TAR и GZ
            elif lower_f.endswith(('.tar', '.tar.gz', '.tgz', '.targz')):
                mode = 'r'
                if '.gz' in lower_f or '.tgz' in lower_f:
                    mode = 'r:gz'
                try:
                    with tarfile.open(path, mode) as tf:
                        tf.extractall(TEMP_DIR)
                except (tarfile.TarError, EOFError, FileNotFoundError):
                    print(f"Ошибка чтения TAR/GZ-архива: {path}")

            # 7z-архивы
            elif lower_f.endswith('.7z'):
                try:
                    with py7zr.SevenZipFile(path, mode='r') as z:
                        z.extractall(TEMP_DIR)
                except py7zr.exceptions.ArchiveError:
                    print(f"Ошибка чтения 7z-архива: {path}")

            # RAR-архивы
            elif lower_f.endswith('.rar'):
                try:
                    with rarfile.RarFile(path) as rf:
                        rf.extractall(TEMP_DIR)
                except rarfile.RarError:
                    print(f"Ошибка чтения RAR-архива: {path}")

    def convert_single_docx_in_temp(self, filename):
        """Конвертирует выбранный .docx сохраняя его в .txt с кодировкой UTF-8."""
        doc_path = os.path.join(TEMP_DIR, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(TEMP_DIR, txt_filename)

        # Читаем .docx
        doc = Document(doc_path)
        full_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)

        # Записываем в .txt с кодировкой UTF-8
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(full_text)

        return txt_path

    def _convert_all_docx_in_temp(self):
        """Конвертирует в temp_dir все .docx сохраняя их в .txt с кодировкой UTF-8."""

        # Получаем список всех .docx файлов
        docx_files = [f for f in os.listdir(TEMP_DIR) if f.lower().endswith(".docx")]

        # Оборачиваем итерацию в tqdm для прогресс-бара
        for filename in tqdm(docx_files, desc="Конвертация DOCX → TXT", unit="файл"):
            # Перебираем все .docx файлы в temp_dir
            if filename.lower().endswith(".docx"):
                _ = self.convert_single_docx_in_temp(filename)

    def _remove_non_txt(self):
        """
        Удаляет в temp_dir все файлы, кроме '.txt'
        """
        allowed_extensions = ('.txt')

        # Получаем список всех .docx файлов
        docx_files = [f for f in os.listdir(TEMP_DIR) if f.lower().endswith(".docx")]

        # Оборачиваем итерацию в tqdm для прогресс-бара
        for f in tqdm(docx_files, desc="Удаление лишних файлов", unit="файл"):
            if not f.lower().endswith(allowed_extensions):
                os.remove(os.path.join(TEMP_DIR, f))

    def _get_text_paths(self) -> list[str]:
        """
        Возвращает список путей ко всем текстовым файлам, доступным после обработки:
        - txt из input,
        - извлечённые из архивов,
        - конвертированные из .docx.
        
        Выход:
            list[str]: Список путей к текстовым файлам.
        """
        text_paths = []
        for f in os.listdir(TEMP_DIR):
            if f.lower().endswith(('.txt')):
                text_paths.append(os.path.join(TEMP_DIR, f))
        return text_paths
