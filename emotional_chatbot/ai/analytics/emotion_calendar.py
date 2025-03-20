# emotion_calendar.py
import random
import calendar
import json
from datetime import datetime
from typing import List, Dict

# Simulación de una base de datos en memoria
emotion_database = {}

def get_daily_emotions(user_id: int, year: int, month: int, day: int) -> List[int]:
    """
    Obtiene la lista de emociones para un usuario en un día específico.
    Retorna una lista de emociones (1-5) para ese día.
    """
    key = (user_id, year, month, day)
    return emotion_database.get(key, [])

def get_daily_average_emotion(user_id: int, year: int, month: int, day: int) -> float:
    """
    Calcula la emoción promediada para un usuario en un día específico.
    Retorna el promedio de las emociones (1-5) para ese día.
    """
    emotions = get_daily_emotions(user_id, year, month, day)
    if not emotions:
        return 0
    return sum(emotions) / len(emotions)

def add_emotion(user_id: int, year: int, month: int, day: int, emotion: int):
    """
    Agrega una emoción para un usuario en un día específico.
    """
    key = (user_id, year, month, day)
    if key not in emotion_database:
        emotion_database[key] = []
    emotion_database[key].append(emotion)

def get_weekly_emotions(user_id: int, year: int, month: int, week: int) -> Dict[int, List[int]]:
    """
    Obtiene las emociones para un usuario en una semana específica.
    Retorna un diccionario con los días de la semana como llaves y listas de emociones como valores.
    """
    cal = calendar.monthcalendar(year, month)
    week_days = cal[week - 1]  # Las semanas comienzan desde 0
    emotions = {day: get_daily_emotions(user_id, year, month, day) for day in week_days if day != 0}
    return emotions

def get_monthly_emotions(user_id: int, year: int, month: int) -> Dict[int, List[int]]:
    """
    Obtiene las emociones para un usuario en un mes específico.
    Retorna un diccionario con los días como llaves y listas de emociones como valores.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    emotions = {day: get_daily_emotions(user_id, year, month, day) for day in range(1, days_in_month + 1)}
    return emotions

def get_yearly_emotions(user_id: int, year: int) -> Dict[int, Dict[int, List[int]]]:
    """
    Obtiene las emociones para un usuario en un año específico.
    Retorna un diccionario con los meses como llaves y un diccionario de emociones por día como valores.
    """
    emotions = {}
    for month in range(1, 13):
        emotions[month] = get_monthly_emotions(user_id, year, month)
    return emotions

def get_emotional_records(user_id: int, year: int, month: int) -> Dict[str, List[int]]:
    """
    Retorna un diccionario con fechas (en formato ISO) como llaves y listas de emociones como valores.
    Este formato es compatible con el widget de Flutter.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    emotional_records = {}
    for day, emotion_list in emotions.items():
        date = datetime(year, month, day).isoformat()  # Formato ISO: YYYY-MM-DD
        emotional_records[date] = emotion_list
    return emotional_records

def save_emotion(user_id: int, date: str, emotion: int):
    """
    Guarda una emoción para un usuario en una fecha específica.
    """
    date_obj = datetime.fromisoformat(date)
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    add_emotion(user_id, year, month, day, emotion)
    return {"status": "success", "message": "Emoción guardada correctamente"}

def query_emotion_data(request_json: str):
    """
    Recibe una consulta en formato JSON y retorna las emociones del usuario para el periodo solicitado.
    """
    try:
        request_data = json.loads(request_json)
        user_id = request_data["user_id"]
        year = request_data.get("year", datetime.now().year)
        month = request_data.get("month", datetime.now().month)
        day = request_data.get("day")
        week = request_data.get("week")

        if day:
            return {
                "emotions": get_daily_emotions(user_id, year, month, day),
                "average_emotion": get_daily_average_emotion(user_id, year, month, day)
            }
        elif week:
            return get_weekly_emotions(user_id, year, month, week)
        elif month:
            return get_monthly_emotions(user_id, year, month)
        else:
            return get_yearly_emotions(user_id, year)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {"error": f"Invalid request: {str(e)}"}

# Estructura de consulta JSON
QUERY_STRUCTURE = {
    "user_id": 123,
    "year": 2025,
    "month": 6,
    "day": 15,
    "week": 2
}