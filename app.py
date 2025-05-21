from flask import Flask, request, jsonify
import joblib
import pandas as pd
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('best_rf_model_fm.pkl')  # Ensure this model file is in the same directory as the app.py

@app.route('/')
def home():
    return "Welcome to the Wildfire Prediction API!"

# API to predict the class
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the user (in JSON format)
        data = request.get_json()

        # Convert input to DataFrame (replace with your feature names and preprocessing as necessary)
        input_data = pd.DataFrame(data, index=[0])

        # Predict using the trained model
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

# Function to run the Flask app in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent the server from restarting multiple times

# Start Flask in a new thread
if __name__ == "__main__":
    thread = threading.Thread(target=run_flask)
    thread.start()

