from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Загружаем данные...")
# Загрузка данных
train = pd.read_csv('source/unzip/train.csv')
test = pd.read_csv('source/unzip/test.csv')
stores = pd.read_csv('source/unzip/stores.csv')
oil = pd.read_csv('source/unzip/oil.csv')
holidays = pd.read_csv('source/unzip/holidays_events.csv')
transactions = pd.read_csv('source/unzip/transactions.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Преобразуем даты в формат datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays['date'] = pd.to_datetime(holidays['date'])
transactions['date'] = pd.to_datetime(transactions['date'])

# Объединяем данные
print("Объединяем данные...")
train_merged = train.merge(stores, on='store_nbr', how='left')
test_merged = test.merge(stores, on='store_nbr', how='left')

train_merged = train_merged.merge(oil, on='date', how='left')
test_merged = test_merged.merge(oil, on='date', how='left')

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
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

# ДОБАВЛЯЕМ СПЕЦИФИЧНЫЕ ДЛЯ ЭКВАДОРА ПРИЗНАКИ
def add_ecuador_features(df):
    """Добавляем признаки специфичные для Эквадора"""
    # Дни выплаты зарплат (15 число и конец месяца)
    df['is_payday'] = ((df['day'] == 15) | (df['is_month_end'] == 1)).astype(int)
    
    # Землетрясение и период после него
    earthquake_date = pd.to_datetime('2016-04-16')
    df['days_after_earthquake'] = (df['date'] - earthquake_date).dt.days
    df['earthquake_effect'] = (
        (df['days_after_earthquake'] >= 0) & 
        (df['days_after_earthquake'] <= 60)  # 2 месяца после землетрясения
    ).astype(int)
    
    return df

print("Добавляем признаки...")
# Применяем функции
train_merged = add_holiday_features(train_merged, holidays)
test_merged = add_holiday_features(test_merged, holidays)

train_merged = add_date_features(train_merged)
test_merged = add_date_features(test_merged)

train_merged = add_ecuador_features(train_merged)
test_merged = add_ecuador_features(test_merged)

# ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print("Обрабатываем пропущенные значения...")

# Заполняем пропуски в dcoilwtico
oil_mean = train_merged['dcoilwtico'].mean()
train_merged['dcoilwtico'] = train_merged['dcoilwtico'].fillna(oil_mean)
test_merged['dcoilwtico'] = test_merged['dcoilwtico'].fillna(oil_mean)

# Заполняем пропуски в transactions
train_merged['transactions'] = train_merged['transactions'].fillna(0)
test_merged['transactions'] = test_merged['transactions'].fillna(0)

# Заполняем пропуски в onpromotion
train_merged['onpromotion'] = train_merged['onpromotion'].fillna(0)
test_merged['onpromotion'] = test_merged['onpromotion'].fillna(0)

# КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
print("Кодируем категориальные признаки...")
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

# ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
print("Подготавливаем данные для модели...")

# Важно: обрабатываем отрицательные продажи (заменяем на 0)
train_merged['sales'] = train_merged['sales'].clip(lower=0)

# Выбираем признаки для обучения
features_to_drop = ['date', 'sales', 'id', 'family', 'city', 'state', 'type', 'days_after_earthquake']
feature_columns = [col for col in train_merged.columns if col not in features_to_drop]

print(f"Используем {len(feature_columns)} признаков:")
for i, col in enumerate(feature_columns):
    print(f"{i+1:2d}. {col}")

# Подготавливаем данные
X_train = train_merged[feature_columns]
y_train = train_merged['sales']
X_test = test_merged[feature_columns]

# Проверяем на пропуски
print(f"\nПроверка данных:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Пропуски в X_train: {X_train.isnull().sum().sum()}")
print(f"Пропуски в X_test: {X_test.isnull().sum().sum()}")

# Заполняем оставшиеся пропуски нулями
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ОБУЧЕНИЕ МОДЕЛИ
print("\n" + "="*60)
print("ОБУЧАЕМ RANDOM FOREST")
print("="*60)

# Создаем модель с оптимизированными параметрами
rf = RandomForestRegressor(
    n_estimators=100,       # количество деревьев
    max_depth=15,           # увеличиваем глубину для лучшего качества
    min_samples_split=10,   # минимум сэмплов для разделения
    min_samples_leaf=5,     # минимум сэмплов в листе
    random_state=42,        # для воспроизводимости
    n_jobs=-1,             # используем все процессоры
    verbose=1              # показываем прогресс
)

print("Начинаем обучение...")
rf.fit(X_train, y_train)
print("Обучение завершено!")

# АНАЛИЗ МОДЕЛИ
print("\n" + "="*60)
print("АНАЛИЗ МОДЕЛИ")
print("="*60)

# 1. Важность признаков
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("ТОП-15 самых важных признаков:")
print(feature_importance.head(15).to_string(index=False))

# 2. Оценка качества на тренировочных данных
train_predictions = rf.predict(X_train)
train_score = rf.score(X_train, y_train)

print(f"\nОценка качества:")
print(f"R² на тренировочных данных: {train_score:.4f}")

# 3. Статистика продаж
print(f"\nСтатистика продаж:")
print(f"Среднее значение продаж: {y_train.mean():.2f}")
print(f"Медиана продаж: {y_train.median():.2f}")
print(f"Максимум продаж: {y_train.max():.2f}")
print(f"Минимум продаж: {y_train.min():.2f}")

# 4. Прогнозы на тестовых данных
print("\nДелаем прогнозы на тестовых данных...")
test_predictions = rf.predict(X_test)

# Убеждаемся, что прогнозы неотрицательные
test_predictions = np.clip(test_predictions, 0, None)

print(f"\nСтатистика прогнозов:")
print(f"Среднее: {test_predictions.mean():.2f}")
print(f"Медиана: {np.median(test_predictions):.2f}")
print(f"Минимум: {test_predictions.min():.2f}")
print(f"Максимум: {test_predictions.max():.2f}")

# ПОДГОТОВКА РЕЗУЛЬТАТА ДЛЯ ОТПРАВКИ
print("\n" + "="*60)
print("ПОДГОТОВКА РЕЗУЛЬТАТА")
print("="*60)

submission = pd.DataFrame({
    'id': test_merged['id'],
    'sales': test_predictions
})

# Проверяем корректность submission
print(f"Размер submission: {submission.shape}")
print(f"Пропуски в submission: {submission.isnull().sum().sum()}")
print(f"Отрицательные значения: {(submission['sales'] < 0).sum()}")

# Сохраняем результат
submission.to_csv('submission.csv', index=False)
print(f"\nРезультат сохранен в 'submission.csv'")

# Показываем первые несколько строк
print("\nПервые 10 строк submission:")
print(submission.head(10))

print("\n" + "="*60)
print("ГОТОВО! Файл submission.csv готов для отправки на Kaggle")
print("="*60)
