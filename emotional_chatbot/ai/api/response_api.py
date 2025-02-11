import os
from flask import Flask, request, jsonify
from transformers import pipeline

# Inicializar Flask y el modelo
app = Flask(__name__)

# Cargar modelo de generación de texto
try:
    chatbot = pipeline("text-generation", model="gpt2")  
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    chatbot = None

@app.route('/chat', methods=['POST'])
def chat():
    if chatbot is None:
        return jsonify({"error": "El modelo no se cargó correctamente"}), 500

    try:
        # Obtener mensaje del usuario
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "No se proporcionó un mensaje"}), 400
        
        # Generar respuesta
        response = chatbot(user_message, max_length=50, num_return_sequences=1)
        bot_response = response[0]['generated_text'].strip()
        
        return jsonify({"response": bot_response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
