import joblib
import numpy as np

MODEL_PATH = "ai/models/response_generator.pkl"

def load_response_model():
    model, vectorizer = joblib.load(MODEL_PATH)
    return model, vectorizer

def generate_response(user_message):
    model, vectorizer = load_response_model()
    X_input = vectorizer.transform([user_message])
    response = model.predict(X_input)[0]
    return response
