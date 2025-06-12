# Прогнозирование продаж в сети магазинов

## Описание данных

### 1. train.csv
Основной обучающий датасет с историческими данными о продажах.

**Структура:**
```
id,date,store_nbr,family,sales,onpromotion
train_0,2013-01-01,1,GROCERY I,0.0,0
train_1,2013-01-01,1,BEAUTY,0.0,0
train_2,2013-01-01,1,PRODUCE,0.0,0
```

**Описание полей:**
- `id` - уникальный идентификатор строки
- `date` - дата продажи (ГГГГ-ММ-ДД)
- `store_nbr` - номер магазина
- `family` - категория товара
- `sales` - объем продаж (целевая переменная)
- `onpromotion` - количество товаров в акции

### 2. test.csv
Тестовый датасет с данными для прогнозирования.

**Структура:**
```
id,date,store_nbr,family,onpromotion
test_0,2017-08-16,1,AUTOMOTIVE,0
test_1,2017-08-16,1,BABY CARE,0
test_2,2017-08-16,1,BEAUTY,0
```

### 3. stores.csv
Информация о магазинах.

**Структура:**
```
store_nbr,city,state,type,cluster
1,Quito,Pichincha,A,13
2,Santo Domingo de los Tsachilas,Santo Domingo de los Tsachilas,C,13
3,Cayambe,Pichincha,C,8
```

**Описание полей:**
- `store_nbr` - номер магазина
- `city` - город
- `state` - провинция
- `type` - тип магазина (A, B, C, D, E)
- `cluster` - группа похожих магазинов

### 4. oil.csv
Исторические данные о цене на нефть.

**Структура:**
```
date,dcoilwtico
2013-01-01,
2013-01-02,93.14
2013-01-03,92.97
```

**Описание полей:**
- `date` - дата
- `dcoilwtico` - цена на нефть (USD/баррель)

### 5. holidays_events.csv
Календарь праздников и событий.

**Структура:**
```
date,type,locale,locale_name,description,transferred
2012-03-02,Holiday,Local,M1,"Fundacion de Manta",False
2012-04-01,Holiday,Regional,Cotopaxi,"Provincializacion de Cotopaxi",False
2012-04-12,Holiday,Local,M2,"Fundacion de Cuenca",False
```

**Описание полей:**
- `date` - дата события
- `type` - тип события (Holiday, Event, etc.)
- `locale` - масштаб (National, Regional, Local)
- `locale_name` - название региона/города
- `description` - описание события
- `transferred` - был ли праздник перенесен

### 6. transactions.csv
Количество транзакций по магазинам.

**Структура:**
```
date,store_nbr,transactions
2013-01-01,25,770
2013-01-02,1,2111
2013-01-02,2,2358
```

**Описание полей:**
- `date` - дата
- `store_nbr` - номер магазина
- `transactions` - количество транзакций

### 7. sample_submission.csv
Пример файла для отправки решений.

**Структура:**
```
id,sales
test_0,0
test_1,0
test_2,0
```

## Источник данных
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
