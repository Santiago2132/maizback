# statistics.py
import matplotlib.pyplot as plt
from collections import Counter
from emotion_calendar import get_monthly_emotions, get_weekly_emotions, get_yearly_emotions, get_daily_emotion

def generate_emotion_statistics(user_id: int, year: int, month: int):
    """
    Genera un gráfico de barras con la distribución de emociones en un mes.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    emotion_counts = Counter(emotions.values())
    
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'orange', 'yellow', 'green', 'blue'])
    plt.xlabel('Emoción')
    plt.ylabel('Frecuencia')
    plt.title(f'Estadísticas de emociones - {month}/{year}')
    plt.xticks(range(1, 6))
    plt.show()

def generate_weekly_emotion_statistics(user_id: int, year: int, month: int, week: int):
    """
    Genera un gráfico de barras con la distribución de emociones en una semana.
    """
    emotions = get_weekly_emotions(user_id, year, month, week)
    emotion_counts = Counter(emotions.values())
    
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'orange', 'yellow', 'green', 'blue'])
    plt.xlabel('Emoción')
    plt.ylabel('Frecuencia')
    plt.title(f'Estadísticas de emociones - Semana {week} de {month}/{year}')
    plt.xticks(range(1, 6))
    plt.show()

def generate_yearly_emotion_statistics(user_id: int, year: int):
    """
    Genera un gráfico de barras con la distribución de emociones en un año.
    """
    emotions = get_yearly_emotions(user_id, year)
    emotion_counts = Counter([emotion for month in emotions.values() for emotion in month.values()])
    
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'orange', 'yellow', 'green', 'blue'])
    plt.xlabel('Emoción')
    plt.ylabel('Frecuencia')
    plt.title(f'Estadísticas de emociones - {year}')
    plt.xticks(range(1, 6))
    plt.show()

def get_emotion_data(user_id: int, year: int, month: int):
    """
    Retorna los datos usados en la gráfica en formato de diccionario.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    return dict(Counter(emotions.values()))

def get_weekly_emotion_data(user_id: int, year: int, month: int, week: int):
    """
    Retorna los datos de emociones de una semana en formato de diccionario.
    """
    emotions = get_weekly_emotions(user_id, year, month, week)
    return dict(Counter(emotions.values()))

def get_yearly_emotion_data(user_id: int, year: int):
    """
    Retorna los datos de emociones de un año en formato de diccionario.
    """
    emotions = get_yearly_emotions(user_id, year)
    return dict(Counter([emotion for month in emotions.values() for emotion in month.values()]))
