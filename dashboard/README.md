# Fraud Detection Dashboard

This folder provides a web-based dashboard for analyzing fraud data and making predictions using machine learning models. It consists of a Flask API for model serving and a Dash application for interactive visualizations.

## Project Structure

```
project_root_folder/
├── model_api/
│   ├── models/
│   │   ├── DecisionTree_Fraud.joblib  
│   │   └── RNN_Credit.pt              
│   └── serve_model.py                 
├── dashboard/        
│   ├── dashboard_app.py               
│   └── requirements.txt               
├── Dockerfile                         
└── README.md                          
```

## Requirements

Install the necessary dependencies:

```bash
pip install -r dashboard/requirements.txt
```

## Running the Application

1. Start the Flask API:
   ```bash
   python model_api/serve_model.py
   ```
2. Start the Dash application:
   ```bash
   python dashboard/dashboard_app.py
   ```
3. Access the dashboard at: [http://127.0.0.1:8050/dashboard/](http://127.0.0.1:8050/dashboard/)

## API Endpoints

- **Home**: `GET /` - Displays a message confirming the API is running.
- **Fraud Prediction**: `POST /predict/fraud` - Predicts fraud based on provided features.



