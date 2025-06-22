# Emotional Chatbot

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/emotional-chatbot.git
   ```
2. Navigate to the project directory:
   ```
   cd emotional-chatbot
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the emotion classification model:
   ```
   python model_training.py
   ```
2. Train the advanced response generation model:
   ```
   python advanced_training.py
   ```
3. Run the chatbot:
   ```
   python emotion_analyzer.py
   ```
4. Interact with the chatbot by entering text when prompted.

## API

The project includes a Flask-based API for the chatbot functionality. To run the API server:

1. Start the API server:
   ```
   python response_api.py
   ```
2. Send a POST request to the `/chat` endpoint with a JSON payload containing the user's message:
   ```
   {
     "message": "Hello, how are you?"
   }
   ```
3. The API will respond with a JSON object containing the chatbot's response.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Submit a pull request to the original repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

To run the tests:

1. Ensure you have the required dependencies installed.
2. Run the test script:
   ```
   python test.py
   ```
