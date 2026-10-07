import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1. DATASET CREATION (Создание датасета)
dates = ['01.10', '02.10', '03.10', '04.10', '05.10']
temps = [12.5, 10.2, np.nan, 9.1, 6.4]
winds = [0.5, 4.5, 0.2, np.nan, 1.1]
pressures = [758, 748, 760, 746, 754]
nmu = [1, 0, 1, 0, 0]

df = pd.DataFrame({
    'date': dates,
    'temperature': temps,
    'wind_speed': winds,
    'pressure': pressures,
    'nmu_status': nmu
})

# 2. DATA CLEANING (Очистка данных от пропусков NaN)
df_clean = df.copy()
df_clean['temperature'] = df_clean['temperature'].fillna(df_clean['temperature'].mean())
df_clean['wind_speed'] = df_clean['wind_speed'].fillna(df_clean['wind_speed'].mean())

# 3. FEATURES AND TARGET (Выделение фич и целевой переменной)
X = df_clean.drop(columns=['date', 'nmu_status'])
y = df_clean['nmu_status']

# 4. MODEL TRAINING (Обучение базовой модели ИИ)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("🤖 Система: 'Первый прототип модели успешно обучен!'")
