from flask import Flask, request, jsonify, send_from_directory
import torch
import joblib
import torch.nn.functional as F
import numpy as np
from model_definitions import  RNNModel  # Import model classes

# Initialize Flask app
app = Flask(__name__)

input_size = 100  # Adjust as per the input feature size of your models

# Load the fraud model
# fraud_model = MLPModel(input_size)
# fraud_model = torch.load('model_api/models/MLP_Fraud.pt')
# fraud_model.eval()
fraud_model = joblib.load('model_api/models/DecisionTree_Fraud.joblib')
# Load the credit card model
creditcard_model = RNNModel(input_size)
creditcard_model = torch.load('model_api/models/RNN_Credit.pt')
creditcard_model.eval()

# Route to serve the favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

# Home route
@app.route('/')
def home():
    return "Model API for Fraud and Credit Card Detection is running!"

# Prediction endpoint for the fraud model
@app.route('/predict/fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)  # Ensure input format
    prediction = fraud_model.predict(features)  # Predict using the loaded joblib model
    return jsonify({'prediction': prediction[0]})

# Prediction endpoint for the credit card model
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
