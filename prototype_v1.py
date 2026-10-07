import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

print("=== ШАГ 1: СОЗДАНИЕ И ОЧИСТКА ДАТАСЕТА ===")
# Задаем метеопараметры через функцию list(), чтобы обойти баги отображения
dates = list(('01.10', '02.10', '03.10', '04.10', '05.10'))
temps = list((12.5, 10.2, np.nan, 9.1, 6.4))
winds = list((0.5, 4.5, 0.2, np.nan, 1.1))

# Цифры давления и статуса НМУ для Красноярска:
pressures = list((758, 748, 760, 746, 754))
nmu = list((1, 0, 1, 0, 0))

df = pd.DataFrame({
    'date': dates,
    'temperature': temps,
    'wind_speed': winds,
    'pressure': pressures,
    'nmu_status': nmu
})

# Автоматически лечим пропуски средними значениями
df_clean = df.copy()
df_clean['temperature'] = df_clean['temperature'].fillna(df_clean['temperature'].mean())
df_clean['wind_speed'] = df_clean['wind_speed'].fillna(df_clean['wind_speed'].mean())
print("Данные успешно очищены от пропусков!")

print("\n=== ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ ИИ ===")
X = df_clean.drop(columns=list(('date', 'nmu_status')))
y = df_clean['nmu_status']

# Создаем и обучаем Случайный лес
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("🤖 ИИ: 'Я успешно обучился на исторических данных!'")

print("\n=== ШАГ 3: ПРОВЕРКА ПРОТОТИПА (ТЕСТ-ДРАЙВ) ===")
# Задаем 3 новых тестовых дня с разной погодой
new_days = {
    'temperature': list((-5.0, 15.0, -2.0)), 
    'wind_speed':  list((0.2, 6.5, 0.3)), 
    'pressure':    list((765.0, 740.0, 758.0))  
}
X_new = pd.DataFrame(new_days)

# Заставляем ИИ сделать прогноз
predictions = model.predict(X_new)

# Сравниваем прогноз с реальным статусом НМУ в эти дни
real_status = list((1, 0, 0))

draft_report = pd.DataFrame({
    'Режим погоды': list(('Мороз + Штиль', 'Тепло + Ветер', 'Холод + Штиль')),
    'Прогноз ИИ (НМУ)': predictions,
    'Реальность': real_status
})
draft_report['Ошибка модели?'] = draft_report['Прогноз ИИ (НМУ)'] != draft_report['Реальность']

# Выводим таблицу ошибок на экран
print(draft_report.to_string(index=False))

print("\n=== ШАГ 4: РАСЧЕТ ВАЖНОСТИ ПРИЗНАКОВ ===")
importances = model.feature_importances_
features_df = pd.DataFrame({
    'Параметр': X.columns,
    'Важность (%)': importances * 100
}).sort_values(by='Важность (%)', ascending=False)

print(features_df.round(2).to_string(index=False))
