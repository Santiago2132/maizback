import os
from flask import Flask, request, jsonify
from main import initialize_system, generate_response

# Inicialización de modelos y datos
models, offensive_words, response_map, patterns_by_intent = initialize_system()

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port, debug=True)