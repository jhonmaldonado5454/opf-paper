import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración inicial
start_date = datetime(2023, 1, 1)
end_date = start_date + timedelta(days=30)
interval = timedelta(minutes=5)

# Lista para guardar los datos
rows = []

# Simulación de feriados (domingos o cada 7 días)
feriados = set()
for i in range(0, 30, 7):
    feriados.add((start_date + timedelta(days=i)).date())

# Generar las filas
current = start_date
while current < end_date:
    hour = current.hour
    minute = current.minute
    day_of_week = current.weekday()
    is_holiday = int(current.date() in feriados)

    # Simular demanda con un patrón base
    base = 100 + 20 * np.sin((hour + minute/60) * np.pi / 12)  # curva día-noche
    fluct = np.random.normal(0, 5)  # ruido
    holiday_penalty = -10 if is_holiday else 0
    demand = base + fluct + holiday_penalty

    rows.append({
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "is_holiday": is_holiday,
        "demand": demand
    })
    current += interval

# Crear DataFrame
df = pd.DataFrame(rows)

# Crear la columna target: demanda 5 minutos después
df["demand_next"] = df["demand"].shift(-1)

# Eliminar la última fila que no tiene target
df = df.dropna().reset_index(drop=True)

# Guardar
df.to_csv("synthetic_demand_5min.csv", index=False)
print(df.head())
