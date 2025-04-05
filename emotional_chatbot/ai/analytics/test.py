import json
from emotion_calendar import query_emotion_data, add_emotion
from statistics import generate_emotion_statistics, get_emotion_data

# Agregar algunas emociones de prueba
add_emotion(123, 2025, 6, 15, 3)
add_emotion(123, 2025, 6, 15, 4)
add_emotion(123, 2025, 6, 16, 5)

# Consultas JSON
request_json_day = json.dumps({"user_id": 123, "year": 2025, "month": 6, "day": 15})
request_json_month = json.dumps({"user_id": 123, "year": 2025, "month": 6})

# Probar funciones
print("Emoción del día:", query_emotion_data(request_json_day))
print("Emociones del mes:", query_emotion_data(request_json_month))

# Generar estadísticas
generate_emotion_statistics(123, 2025, 6, use_averages=False)  # Frecuencia de emociones
generate_emotion_statistics(123, 2025, 6, use_averages=True)   # Distribución de promedios
print("Datos de emociones del mes:", get_emotion_data(123, 2025, 6))