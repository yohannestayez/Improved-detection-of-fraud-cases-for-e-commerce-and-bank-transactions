# Import necessary libraries
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../scripts')  # Appending the path to access the scripts folder
from DL_models import * 


# Load PyTorch models
def load_pytorch_model(model_name, model_dir, input_size):
    model_path = f"{model_dir}/{model_name}.pt"
    if "MLP" in model_name:
        model = MLPModel(input_size)
    elif "CNN" in model_name:
        model = CNNModel(input_size)
    elif "RNN" in model_name:
        model = RNNModel(input_size)
    elif "LSTM" in model_name:
        model = LSTMModel(input_size)
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode
    return model

# Load scikit-learn models
def load_sklearn_model(model_name, model_dir):
    model_path = f"{model_dir}/{model_name}.joblib"
    model = joblib.load(model_path)
    return model

# Function to get SHAP values for PyTorch models
def get_shap_values_pytorch(model, X, num_samples=100):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Wrapper function for model predictions in the SHAP-compatible format
    def model_predict(x):
        x_tensor = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = model(x_tensor)  # Get logits
            probabilities = F.softmax(logits, dim=1)  # Apply softmax for probabilities over classes
        return probabilities.numpy()

    # KernelExplainer for SHAP with general model compatibility
    explainer = shap.KernelExplainer(model_predict, X[:num_samples].astype(np.float32))
    shap_values = explainer.shap_values(X[:num_samples].astype(np.float32))
    
    return shap_values


# Function to predict probabilities for binary classification with PyTorch
def pytorch_predict_proba_binary(model, data):
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data).float()
        outputs = model(data_tensor)
        # Check if output is single probability; transform if necessary
        if outputs.shape[1] == 1:  # Binary classification with single output
            probabilities = torch.sigmoid(outputs).cpu().numpy()  # Single class probability
            probabilities = np.hstack((1 - probabilities, probabilities))  # Convert to two-class format
        else:
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities