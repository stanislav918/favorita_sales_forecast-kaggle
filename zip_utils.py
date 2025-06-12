import zipfile
import pandas as pd
from io import StringIO

def read_csv_from_zip(zip_path, csv_name, **kwargs):
    """
    Читает CSV файл из ZIP архива в pandas DataFrame
    
    :param zip_path: путь к ZIP архиву
    :param csv_name: имя CSV файла внутри архива
    :param kwargs: дополнительные параметры для pd.read_csv
    :return: pandas DataFrame
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            content = f.read().decode('utf-8')
            return pd.read_csv(StringIO(content), **kwargs)

def get_zip_file_list(zip_path):
    """
    Возвращает список файлов в ZIP архиве
    
    :param zip_path: путь к ZIP архиву
    :return: список имен файлов
    """
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()
