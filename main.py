from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from zip_utils import read_csv_from_zip

# Путь к архиву с данными
ZIP_PATH = 'source/store-sales-time-series-forecasting.zip'

# Загрузка данных напрямую из архива
print("Загрузка данных из архива...")
train = read_csv_from_zip(ZIP_PATH, 'train.csv')
test = read_csv_from_zip(ZIP_PATH, 'test.csv')
stores = read_csv_from_zip(ZIP_PATH, 'stores.csv')
oil = read_csv_from_zip(ZIP_PATH, 'oil.csv')
holidays = read_csv_from_zip(ZIP_PATH, 'holidays_events.csv')
transactions = read_csv_from_zip(ZIP_PATH, 'transactions.csv')

# Преобразуем даты в формат datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays['date'] = pd.to_datetime(holidays['date'])
transactions['date'] = pd.to_datetime(transactions['date'])

# Объединяем train с stores (по store_nbr)
train_merged = train.merge(stores, on='store_nbr', how='left')
test_merged = test.merge(stores, on='store_nbr', how='left')

# Объединяем с oil (по date)
train_merged = train_merged.merge(oil, on='date', how='left')
test_merged = test_merged.merge(oil, on='date', how='left')

# Объединяем с transactions (по date и store_nbr)
train_merged = train_merged.merge(transactions, on=['date', 'store_nbr'], how='left')
test_merged = test_merged.merge(transactions, on=['date', 'store_nbr'], how='left')

# УЛУЧШЕННАЯ ОБРАБОТКА ПРАЗДНИКОВ
def add_holiday_features(df, holidays_df):
    """Добавляем признаки праздников"""
    # Национальные праздники
    national_holidays = holidays_df[holidays_df['locale'] == 'National'].copy()
    national_holidays['is_national_holiday'] = 1
    df = df.merge(
        national_holidays[['date', 'is_national_holiday']], 
        on='date', 
        how='left'
    )
    df['is_national_holiday'] = df['is_national_holiday'].fillna(0)
    
    # Региональные праздники
    regional_holidays = holidays_df[holidays_df['locale'] == 'Regional'].copy()
    regional_holidays['is_regional_holiday'] = 1
    df = df.merge(
        regional_holidays[['date', 'locale_name', 'is_regional_holiday']], 
        left_on=['date', 'state'], 
        right_on=['date', 'locale_name'], 
        how='left'
    )
    df['is_regional_holiday'] = df['is_regional_holiday'].fillna(0)
    df = df.drop('locale_name', axis=1, errors='ignore')
    
    # Локальные праздники
    local_holidays = holidays_df[holidays_df['locale'] == 'Local'].copy()
    local_holidays['is_local_holiday'] = 1
    df = df.merge(
        local_holidays[['date', 'locale_name', 'is_local_holiday']], 
        left_on=['date', 'city'], 
        right_on=['date', 'locale_name'], 
        how='left'
    )
    df['is_local_holiday'] = df['is_local_holiday'].fillna(0)
    df = df.drop('locale_name', axis=1, errors='ignore')
    
    # Общий признак праздника
    df['is_any_holiday'] = (df['is_national_holiday'] + 
                           df['is_regional_holiday'] + 
                           df['is_local_holiday']).clip(0, 1)
    
    return df

# ДОБАВЛЯЕМ ВРЕМЕННЫЕ ПРИЗНАКИ
def add_date_features(df):
    """Добавляем временные признаки"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    return df

# Применяем функции
train_merged = add_holiday_features(train_merged, holidays)
test_merged = add_holiday_features(test_merged, holidays)

train_merged = add_date_features(train_merged)
test_merged = add_date_features(test_merged)

# ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# Заполняем пропуски в dcoilwtico средним значением
train_merged['dcoilwtico'] = train_merged['dcoilwtico'].fillna(train_merged['dcoilwtico'].mean())
test_merged['dcoilwtico'] = test_merged['dcoilwtico'].fillna(test_merged['dcoilwtico'].mean())

# Заполняем пропуски в transactions нулями (нет данных = нет транзакций)
train_merged['transactions'] = train_merged['transactions'].fillna(0)
test_merged['transactions'] = test_merged['transactions'].fillna(0)

# КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
from sklearn.preprocessing import LabelEncoder

categorical_columns = ['family', 'city', 'state', 'type']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    # Объединяем train и test для consistent encoding
    combined_values = pd.concat([train_merged[col], test_merged[col]]).astype(str)
    le.fit(combined_values)
    
    train_merged[col + '_encoded'] = le.transform(train_merged[col].astype(str))
    test_merged[col + '_encoded'] = le.transform(test_merged[col].astype(str))
    
    label_encoders[col] = le

# Проверяем результат
print("Тренировочные данные после всех преобразований:")
print(f"Форма: {train_merged.shape}")
print("\nСтолбцы:")
print(train_merged.columns.tolist())
print(f"\nПропуски в train: {train_merged.isnull().sum().sum()}")
print(f"Пропуски в test: {test_merged.isnull().sum().sum()}")
