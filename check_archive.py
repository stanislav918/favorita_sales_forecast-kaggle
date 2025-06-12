from zip_utils import get_zip_file_list

ZIP_PATH = 'source/store-sales-time-series-forecasting.zip'

def main():
    print("Проверка содержимого архива...")
    try:
        files = get_zip_file_list(ZIP_PATH)
        print("Доступные файлы в архиве:")
        for file in files:
            print(f"- {file}")
    except FileNotFoundError:
        print(f"Ошибка: Файл {ZIP_PATH} не найден")
    except zipfile.BadZipFile:
        print(f"Ошибка: Файл {ZIP_PATH} не является корректным ZIP архивом")

if __name__ == "__main__":
    main()
