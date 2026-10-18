import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. ИНЖИНИРИНГ ПРИЗНАКОВ (Добавляем эффект штиля)
# Если скорость ветра (столбец 'Ff') от 0 до 1 м/с -> признак равен 1, иначе 0
# (Если в твоей таблице столбец ветра называется по-другому, замени 'Ff')
if 'Ff' in df.columns:
    df['is_calm'] = np.where(df['Ff'] <= 1.0, 1, 0)

# 2. АВТОМАТИЧЕСКИЙ ПОИСК СТОЛБЦА С ДАТОЙ И СОРТИРОВКА
# Ищем столбец, в названии которого есть слово 'время' или 'Date'
date_col = [col for col in df.columns if 'время' in col.lower() or 'date' in col.lower()]

if date_col:
    # Переименовываем найденный столбец в 'date' для удобства
    df = df.rename(columns={date_col[0]: 'date'})
    print(f"Успешно нашли столбец времени: имя '{date_col[0]}' изменено на 'date'")
else:
    # Если вдруг не нашли, выведем все имена, чтобы ты подсказал
    raise KeyError(f"Не нашли столбец времени. Доступные столбцы: {list(df.columns)}")

# Переводим в формат даты и строго сортируем по хронологии
df['date'] = pd.to_datetime(df['date'], dayfirst=True) 
df = df.sort_values('date').reset_index(drop=True)

# 3. ОБУЧЕНИЕ МОДЕЛИ С БАЛАНСИРОВКОЙ КЛАССОВ
# (Вставляем этот блок, который случайно потерялся)
baseline_model = LogisticRegression(class_weight='balanced', random_state=42)
baseline_model.fit(X_train, y_train)

# 4. МЕТРИКИ ОЦЕНКИ
y_pred = baseline_model.predict(X_test)

print("\n=== ОТЧЕТ ПО МЕТРИКАМ БЕЙСЛАЙНА ===")
print(classification_report(y_test, y_pred, target_names=['Чисто', 'НМУ']))
