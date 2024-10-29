from flask import Flask, request, jsonify, send_from_directory
import torch
import joblib
import torch.nn.functional as F
import numpy as np
from model_definitions import RNNModel
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

input_size = 100  

# Load the fraud model
fraud_model = joblib.load('model_api/models/DecisionTree_Fraud.joblib')

# Load the credit card model
creditcard_model = RNNModel(input_size)
creditcard_model = torch.load('model_api/models/RNN_Credit.pt')
creditcard_model.eval()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.route('/')
def home():
    app.logger.info("Home endpoint accessed")
    return "Model API for Fraud and Credit Card Detection is running!"

@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = fraud_model.predict(features)
        
        app.logger.info(f"Fraud prediction request received with data: {data}")
        app.logger.info(f"Fraud prediction result: {prediction[0]}")
        
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error in fraud prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/creditcard', methods=['POST'])
def predict_creditcard():
    try:
        data = request.json['data']
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        with torch.no_grad():
            output = creditcard_model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy().tolist()
        
        app.logger.info(f"Credit card prediction request received with data: {data}")
        app.logger.info(f"Credit card prediction probabilities: {probabilities}")
        
        return jsonify({'creditcard_predictions': probabilities})
    except Exception as e:
        app.logger.error(f"Error in credit card prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
