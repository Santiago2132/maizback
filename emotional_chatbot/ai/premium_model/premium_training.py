import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# Descargar recursos NLTK (ejecutar solo primera vez)
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Configuración inicial de directorios
DIRECTORIES = ["ai/logs", "ai/models2", "ai/metrics", "ai/data"]
for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai/logs/training.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TextPreprocessor:
    """Realiza limpieza y preprocesamiento de texto"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Realiza limpieza completa del texto"""
        text = str(text).lower().strip()
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Lemmatización y eliminación de stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

def load_and_preprocess_data():
    """Carga y preprocesa datos desde intents2.json"""
    try:
        logging.info("Iniciando carga y preprocesamiento de datos...")
        
        # Carga de datos desde intents2.json
        with open("ai/data/intents2.json", "r") as f:
            intents_data = json.load(f)
        
        texts = []
        labels = []
        for intent in intents_data['intents']:
            for pattern in intent['patterns']:
                texts.append(pattern)
                labels.append(intent['tag'])
        
        # Preprocesamiento de texto
        preprocessor = TextPreprocessor()
        texts = [preprocessor.clean_text(text) for text in texts]
        
        # Codificación de etiquetas
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        logging.info(f"Datos procesados: {len(texts)} muestras, {len(le.classes_)} clases")
        
        return texts, encoded_labels, le
    
    except Exception as e:
        logging.error(f"Error en preprocesamiento: {str(e)}")
        raise

def train_emotional_model():
    """Entrena el modelo con validación cruzada y optimización"""
    try:
        start_time = datetime.now()
        logging.info("Iniciando entrenamiento del modelo...")
        
        # Cargar y preprocesar datos
        texts, labels, le = load_and_preprocess_data()
        
        # Validación de datos
        if len(np.unique(labels)) < 2:
            raise ValueError("Se requieren al menos 2 clases para entrenamiento")
        
        # Crear pipeline completo con SMOTE para manejo de clases desbalanceadas
        pipeline = make_imb_pipeline(
            TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3,
                max_df=0.9
            ),
            SMOTE(random_state=42),
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            )
        )
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline,
            texts,
            labels,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Entrenamiento final
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )
        
        pipeline.fit(X_train, y_train)
        
        # Evaluación
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Generar reporte
        class_names = le.inverse_transform(np.unique(y_val))
        report = classification_report(
            y_val,
            y_pred,
            labels=np.unique(y_val),
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Métricas para guardar
        training_stats = {
            "timestamp": start_time.isoformat(),
            "training_time": str(datetime.now() - start_time),
            "cv_scores": {
                "mean_accuracy": round(float(cv_scores.mean()), 4),
                "std_accuracy": round(float(cv_scores.std()), 4),
                "scores": cv_scores.tolist()
            },
            "test_accuracy": float(accuracy),
            "class_distribution": dict(zip(
                class_names,
                np.bincount(y_val).tolist()
            )),
            "classification_report": json.loads(
                json.dumps(report, default=lambda x: float(x) if isinstance(x, (np.int64, np.float64)) else x)
            ),
            "model_metadata": {
                "features": int(len(pipeline.named_steps['tfidfvectorizer'].get_feature_names_out())),
                "classes": class_names.tolist(),
                "parameters": {k: str(v) for k, v in pipeline.get_params().items()}
            }
        }
        
        # Guardado de artefactos
        joblib.dump(pipeline, "ai/models2/emotion_model.pipeline")
        joblib.dump(le, "ai/models2/label_encoder.pkl")
        
        with open("ai/metrics/training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False)
        
        logging.info(
            f"Entrenamiento completado en {training_stats['training_time']}\n"
            f"Validación cruzada Accuracy: {training_stats['cv_scores']['mean_accuracy']:.2%} (±{training_stats['cv_scores']['std_accuracy']:.2%})\n"
            f"Precisión final: {accuracy:.2%}\n"
            f"Clases detectadas: {len(class_names)}"
        )
        
        return training_stats
    
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    train_emotional_model()