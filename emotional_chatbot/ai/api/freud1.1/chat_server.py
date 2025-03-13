import runpy
from flask import Blueprint, request, jsonify

# Definimos el blueprint para las rutas de chat
chat_bp = Blueprint('chat', __name__)

# Cargar response2.py dinámicamente
response2_path = "M:/maizback/emotional_chatbot/ai/training/response2.py"
response2 = runpy.run_path(response2_path)

# Inicialización de modelos y datos
models, offensive_words, response_map, patterns_by_intent = response2["initialize_system"]()
generate_response = response2["generate_response"]

@chat_bp.route('/', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No se proporcionó un mensaje"}), 400
        
        bot_response = generate_response(
            user_input=user_message,
            models=models,
            response_map=response_map,
            patterns_by_intent=patterns_by_intent,
            offensive_words=offensive_words
        )
        
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500