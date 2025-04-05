import json
from typing import List, Dict, Tuple
from datetime import datetime
import calendar

# Cargar datos desde un JSON estático (simulación)
try:
    with open('emotions.json', 'r') as f:
        emotion_database = json.load(f)
except FileNotFoundError:
    # Si no existe el archivo, inicializamos un diccionario vacío
    emotion_database = {}

def save_to_json():
    """Guarda los datos en el archivo JSON."""
    with open('emotions.json', 'w') as f:
        json.dump(emotion_database, f, indent=4)

def get_daily_emotions(user_id: int, year: int, month: int, day: int) -> Tuple[List[int], float]:
    """
    Obtiene la lista de emociones y el promedio para un día específico.
    Retorna una tupla: (lista de emociones, promedio).
    """
    key = (user_id, year, month, day)
    emotions = emotion_database.get(str(key), [])  # Convertimos la clave a string para JSON
    avg = sum(emotions) / len(emotions) if emotions else 0.0
    return emotions, avg

def add_emotion(user_id: int, year: int, month: int, day: int, emotion: int):
    """
    Agrega una emoción a un día específico y guarda en el JSON.
    """
    key = (user_id, year, month, day)
    key_str = str(key)
    if key_str not in emotion_database:
        emotion_database[key_str] = []
    emotion_database[key_str].append(emotion)
    save_to_json()

def get_weekly_emotions(user_id: int, year: int, month: int, week: int) -> Dict[int, Tuple[List[int], float]]:
    """
    Obtiene las emociones y promedios de una semana específica.
    Retorna un diccionario con días como claves y tuplas (emociones, promedio) como valores.
    """
    cal = calendar.monthcalendar(year, month)
    if week < 1 or week > len(cal):
        return {}
    week_days = cal[week - 1]
    emotions = {
        day: get_daily_emotions(user_id, year, month, day)
        for day in week_days if day != 0
    }
    return emotions

def get_monthly_emotions(user_id: int, year: int, month: int) -> Dict[int, Tuple[List[int], float]]:
    """
    Obtiene las emociones y promedios de un mes específico.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    emotions = {
        day: get_daily_emotions(user_id, year, month, day)
        for day in range(1, days_in_month + 1)
    }
    return emotions

def get_yearly_emotions(user_id: int, year: int) -> Dict[int, Dict[int, Tuple[List[int], float]]]:
    """
    Obtiene las emociones y promedios de un año específico.
    """
    emotions = {
        month: get_monthly_emotions(user_id, year, month)
        for month in range(1, 13)
    }
    return emotions

def get_emotional_records(user_id: int, year: int, month: int) -> Dict[str, Dict[str, any]]:
    """
    Retorna un diccionario con fechas en formato ISO, lista de emociones y promedio.
    Compatible con Flutter.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    emotional_records = {}
    for day, (emotion_list, avg) in emotions.items():
        date = datetime(year, month, day).isoformat()
        emotional_records[date] = {"emotions": emotion_list, "average": avg}
    return emotional_records

def save_emotion(user_id: int, date: str, emotion: int):
    """
    Guarda una emoción para una fecha específica.
    """
    date_obj = datetime.fromisoformat(date)
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    add_emotion(user_id, year, month, day, emotion)
    return {"status": "success", "message": "Emoción guardada correctamente"}

def query_emotion_data(request_json: str):
    """
    Procesa una consulta JSON y retorna las emociones y promedios correspondientes.
    """
    try:
        request_data = json.loads(request_json)
        user_id = request_data["user_id"]
        year = request_data.get("year", datetime.now().year)
        month = request_data.get("month", datetime.now().month)
        day = request_data.get("day")
        week = request_data.get("week")

        if day:
            emotions, avg = get_daily_emotions(user_id, year, month, day)
            return {"emotions": emotions, "average_emotion": avg}
        elif week:
            return get_weekly_emotions(user_id, year, month, week)
        elif month:
            return get_monthly_emotions(user_id, year, month)
        else:
            return get_yearly_emotions(user_id, year)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {"error": f"Invalid request: {str(e)}"}