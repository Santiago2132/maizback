# test.py
import json
from emotion_calendar import query_emotion_data
from statistics import generate_emotion_statistics, get_emotion_data, generate_weekly_emotion_statistics, get_weekly_emotion_data, generate_yearly_emotion_statistics, get_yearly_emotion_data

# Ejemplo de consulta JSON para el día
request_json_day = json.dumps({
    "user_id": 123,
    "year": 2025,
    "month": 6,
    "day": 15
})

# Ejemplo de consulta JSON para la semana
request_json_week = json.dumps({
    "user_id": 123,
    "year": 2025,
    "month": 6,
    "week": 2
})

# Ejemplo de consulta JSON para el mes
request_json_month = json.dumps({
    "user_id": 123,
    "year": 2025,
    "month": 6
})

# Ejemplo de consulta JSON para el año
request_json_year = json.dumps({
    "user_id": 123,
    "year": 2025
})

# Obtener emociones del día
emotions_day = query_emotion_data(request_json_day)
print("Emoción del día:", emotions_day)

# Obtener emociones de la semana
emotions_week = query_emotion_data(request_json_week)
print("Emociones de la semana:", emotions_week)

# Obtener emociones del mes
emotions_month = query_emotion_data(request_json_month)
print("Emociones del mes:", emotions_month)

# Obtener emociones del año
emotions_year = query_emotion_data(request_json_year)
print("Emociones del año:", emotions_year)

# Generar estadísticas y mostrar el gráfico del mes
generate_emotion_statistics(123, 2025, 6)

# Generar estadísticas y mostrar el gráfico de la semana
generate_weekly_emotion_statistics(123, 2025, 6, 2)

# Generar estadísticas y mostrar el gráfico del año
generate_yearly_emotion_statistics(123, 2025)

# Obtener datos de emociones en formato de diccionario del mes
emotion_data_month = get_emotion_data(123, 2025, 6)
print("Datos de emociones del mes:", emotion_data_month)

# Obtener datos de emociones en formato de diccionario de la semana
emotion_data_week = get_weekly_emotion_data(123, 2025, 6, 2)
print("Datos de emociones de la semana:", emotion_data_week)

# Obtener datos de emociones en formato de diccionario del año
emotion_data_year = get_yearly_emotion_data(123, 2025)
print("Datos de emociones del año:", emotion_data_year)