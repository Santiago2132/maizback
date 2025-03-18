# emotion_calendar.py
import random
import calendar
import json
from datetime import datetime

def get_daily_emotion(user_id: int, year: int, month: int, day: int):
    """
    Simula la consulta de emociones para un usuario en un día específico.
    Retorna la emoción (1-5) para ese día.
    """
    return random.randint(1, 5)

def get_weekly_emotions(user_id: int, year: int, month: int, week: int):
    """
    Simula la consulta de emociones para un usuario en una semana específica.
    Retorna un diccionario con los días de la semana como llaves y emociones (1-5) como valores.
    """
    cal = calendar.monthcalendar(year, month)
    week_days = cal[week - 1]  # Las semanas comienzan desde 0
    emotions = {day: random.randint(1, 5) for day in week_days if day != 0}
    return emotions

def get_monthly_emotions(user_id: int, year: int, month: int):
    """
    Simula la consulta de emociones para un usuario en un mes específico.
    Retorna un diccionario con los días como llaves y emociones (1-5) como valores.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    emotions = {day: random.randint(1, 5) for day in range(1, days_in_month + 1)}
    return emotions

def get_yearly_emotions(user_id: int, year: int):
    """
    Simula la consulta de emociones para un usuario en un año específico.
    Retorna un diccionario con los meses como llaves y un diccionario de emociones por día como valores.
    """
    emotions = {}
    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
        emotions[month] = {day: random.randint(1, 5) for day in range(1, days_in_month + 1)}
    return emotions

def get_emotional_records(user_id: int, year: int, month: int):
    """
    Retorna un diccionario con fechas (en formato ISO) como llaves y emociones (1-5) como valores.
    Este formato es compatible con el widget de Flutter.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    emotional_records = {}
    for day, emotion in emotions.items():
        date = datetime(year, month, day).isoformat()  # Formato ISO: YYYY-MM-DD
        emotional_records[date] = emotion
    return emotional_records

def save_emotion(user_id: int, date: str, emotion: int):
    """
    Simula guardar una emoción para un usuario en una fecha específica.
    """
    # Aquí podrías implementar la lógica para guardar en una base de datos
    print(f"Emoción guardada: Usuario {user_id}, Fecha {date}, Emoción {emotion}")
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
            return get_daily_emotion(user_id, year, month, day)
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