from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
from model_definitions import MLPModel, RNNModel  # Import model classes

# Initialize Flask app
app = Flask(__name__)

input_size = 30 
# Load the fraud model
fraud_model = MLPModel(input_size)
fraud_model.load_state_dict(torch.load('models/MLP_Fraud.pt'))
fraud_model.eval()

# Load the credit card model
creditcard_model = RNNModel(input_size)
creditcard_model.load_state_dict(torch.load('models/RNN_Credit.pt'))
creditcard_model.eval()

# Define routes
@app.route('/')
def home():
    return "Model API for Fraud and Credit Card Detection is running!"

# Prediction for Fraud model
@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.json['data']
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        with torch.no_grad():
            output = fraud_model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy().tolist()
        
        return jsonify({'fraud_predictions': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)})

# Prediction for Credit Card model
@app.route('/predict/creditcard', methods=['POST'])
def predict_creditcard():
    try:
        data = request.json['data']
        input_tensor = torch.tensor(data, dtype=torch.float32)
        
        with torch.no_grad():
            output = creditcard_model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy().tolist()
        
        return jsonify({'creditcard_predictions': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
