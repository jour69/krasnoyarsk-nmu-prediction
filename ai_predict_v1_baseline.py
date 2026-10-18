import pandas as pd
import numpy as np

filepath = "/krasnoyarsk_meteo_2024_2026.csv"

try:
    df = pd.read_csv(filepath, skiprows=6, sep=';', index_col=False, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(filepath, skiprows=6, sep=';', index_col=False, encoding='cp1251')

df.columns = df.columns.str.strip()
time_col = [col for col in df.columns if 'время' in col.lower()][0]
df = df.rename(columns={time_col: 'datetime'})

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')
df = df.sort_values('datetime').reset_index(drop=True)

features = ['datetime', 'T', 'Po', 'Ff', 'U']
df = df[[col for col in features if col in df.columns]]
df = df.ffill()

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Исторические периоды НМУ Красноярска для разметки
# Полный и академически точный список исторических периодов НМУ для разметки
nmu_periods = [
    # --- 2024 год ---
    ('2024-01-10 00:00', '2024-01-15 23:59'),
    ('2024-02-19 19:00', '2024-02-22 19:00'),
    ('2024-06-12 16:00', '2024-06-17 09:00'),
    ('2024-12-11 00:00', '2024-12-16 23:59'),
    
    # --- 2025 год ---
    ('2025-02-03 00:00', '2025-02-07 23:59'),
    ('2025-02-15 00:00', '2025-02-18 23:59'),
    ('2025-07-24 00:00', '2025-07-28 23:59'),
    ('2025-10-25 19:00', '2025-10-27 19:00'),
    
    # --- 2026 год (Тестовая выборка) ---
    ('2026-01-16 13:00', '2026-01-24 19:00'),
    ('2026-01-26 14:00', '2026-01-28 19:00'),
    ('2026-07-01 19:00', '2026-07-05 09:00'),
    ('2026-07-07 16:00', '2026-07-09 19:00'),
    ('2026-09-22 19:00', '2026-09-24 19:00')
]

df['nmu_status'] = 0
for start, end in nmu_periods:
    df.loc[(df['datetime'] >= pd.to_datetime(start)) & (df['datetime'] <= pd.to_datetime(end)), 'nmu_status'] = 1

train_df = df[df['datetime'] < '2026-01-01'].reset_index(drop=True)
test_df = df[df['datetime'] >= '2026-01-01'].reset_index(drop=True)

print(f"Train (2024-2025): {train_df.shape[0]} строк, НМУ: {train_df['nmu_status'].sum()}")
print(f"Test (2026): {test_df.shape[0]} строк, НМУ: {test_df['nmu_status'].sum()}")
import numpy as np

print("=== СТАРТ ИНЖИНИРИНГ ПРИЗНАКОВ (ИИ-Модуль v1.5) ===")

# 1. КОДИРОВАНИЕ КРИТИЧЕСКОГО ШТИЛЯ (Скорость ветра)
if 'Ff' in df.columns:
    df['danger_wind'] = np.where(df['Ff'] <= 1.5, 1, 0)
    print("✅ Создан признак опасного штиля: 'danger_wind'")
else:
    df['danger_wind'] = 0

# 2. РАСЧЕТ ИНДИКАТОРА ИНВЕРСИИ И ЗАСТОЯ (Через влажность 'U' и ветер)
if 'U' in df.columns and 'danger_wind' in df.columns:
    df['inversion_risk'] = np.where((df['U'] >= 80.0) & (df['danger_wind'] == 1), 1, 0)
    print("✅ Создан признак инверсионного риска: 'inversion_risk'")
else:
    df['inversion_risk'] = 0
    print("❌ Предупреждение: столбец влажности 'U' не найден в df. Проверь Шаг 1!")

# 3. ДОБАВЛЯЕМ СЕЗОННОСТЬ И ВРЕМЯ СУТОК
if 'datetime' in df.columns:
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    
    # Кодируем зиму: 12 (декабрь), 1 (январь), 2 (февраль)
    df['is_winter'] = np.where(df['month'].isin([12, 1, 2]), 1, 0)
    print("✅ Добавлены временные предикторы: 'month', 'hour', 'is_winter'")

print(f"\nВсего ИИ-признаков готово к обучению: {df.shape}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("=== СТАРТ ИИ-ОБУЧЕНИЯ (ИИ-Модуль v1.5) ===")

# 1. РАЗДЕЛЕНИЕ ВРЕМЕННОГО РЯДА
train_features = df[df['datetime'] < '2026-01-01']
test_features = df[df['datetime'] >= '2026-01-01']

# 2. ОБНОВЛЕННЫЙ СПИСОК ИИ-ПРИЗНАКОВ С УЧЕТОМ ИНВЕРСИИ
feature_cols = ['danger_wind', 'inversion_risk', 'is_winter', 'T', 'Po']
target_col = 'nmu_status'

X_train, y_train = train_features[feature_cols], train_features[target_col].astype(int)
X_test, y_test = test_features[feature_cols], test_features[target_col].astype(int)

print(f"Обучающая выборка (Train): {X_train.shape} строк, точек НМУ: {y_train.sum()}")
print(f"Тестовая выборка (Test): {X_test.shape} строк, точек НМУ: {y_test.sum()}")

# 3. ОБУЧЕНИЕ МОДЕЛИ С ИИ-БАЛАНСИРОВКОЙ
ai_baseline = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
ai_baseline.fit(X_train, y_train)

# 4. РАСЧЕТ И ВЫВОД ОБНОВЛЕННЫХ МЕТРИК
y_pred = ai_baseline.predict(X_test)

print("\n=== ОБНОВЛЕННЫЕ МЕТРИКИ ИИ (ИИ-Модуль v1.5) ===")
print(classification_report(y_test, y_pred, target_names=['Чисто', 'НМУ']))
