from typing import Dict
import matplotlib.pyplot as plt
from collections import Counter
from emotion_calendar import get_monthly_emotions, get_weekly_emotions, get_yearly_emotions

def generate_emotion_statistics(user_id: int, year: int, month: int, use_averages: bool = False):
    """
    Genera un gráfico de barras con la distribución de emociones o promedios en un mes.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    if use_averages:
        averages = [avg for _, avg in emotions.values() if avg > 0]
        plt.hist(averages, bins=5, range=(1, 5), color='blue', edgecolor='black')
        plt.xlabel('Promedio de Emoción')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de Promedios - {month}/{year}')
    else:
        all_emotions = [emotion for emotions_list, _ in emotions.values() for emotion in emotions_list]
        emotion_counts = Counter(all_emotions)
        plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'orange', 'yellow', 'green', 'blue'])
        plt.xlabel('Emoción')
        plt.ylabel('Frecuencia')
        plt.title(f'Estadísticas de Emociones - {month}/{year}')
        plt.xticks(range(1, 6))
    plt.show()

def generate_weekly_emotion_statistics(user_id: int, year: int, month: int, week: int, use_averages: bool = False):
    """
    Genera un gráfico de barras con la distribución de emociones o promedios en una semana.
    """
    emotions = get_weekly_emotions(user_id, year, month, week)
    if use_averages:
        averages = [avg for _, avg in emotions.values() if avg > 0]
        plt.hist(averages, bins=5, range=(1, 5), color='blue', edgecolor='black')
        plt.xlabel('Promedio de Emoción')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de Promedios - Semana {week} de {month}/{year}')
    else:
        all_emotions = [emotion for emotions_list, _ in emotions.values() for emotion in emotions_list]
        emotion_counts = Counter(all_emotions)
        plt.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'orange', 'yellow', 'green', 'blue'])
        plt.xlabel('Emoción')
        plt.ylabel('Frecuencia')
        plt.title(f'Estadísticas de Emociones - Semana {week} de {month}/{year}')
        plt.xticks(range(1, 6))
    plt.show()

def get_emotion_data(user_id: int, year: int, month: int) -> Dict[int, int]:
    """
    Retorna las frecuencias de emociones individuales en un mes.
    """
    emotions = get_monthly_emotions(user_id, year, month)
    all_emotions = [emotion for emotions_list, _ in emotions.values() for emotion in emotions_list]
    return dict(Counter(all_emotions))