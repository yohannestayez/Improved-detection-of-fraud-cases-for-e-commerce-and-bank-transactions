# Improved detection of fraud cases for e-commerce and bank transactions

## overview
This repository includes a Jupyter notebook that demonstrates a complete workflow for data analysis, preprocessing, and model interpretability. It guides users through preparing datasets for machine learning applications, particularly in credit scoring and fraud detection. The notebook also includes code for training, saving, and interpreting both deep learning models and traditional machine learning classifiers. Using explainability techniques like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations), this project emphasizes model transparency to build trust and enhance decision-making insights.

## Notebooks

### Data_Analysis_and_Preprocessing.ipynb
- This notebook demonstrates how to handle raw data, clean it, perform exploratory data analysis (EDA), and transform it into a format suitable for machine learning algorithms. 

- **Key Features**:
    - **Loading the Data**: Reading raw data from various sources such as CSV, JSON, or Excel files.
    -  **Data Cleaning**: Handling missing values, correcting data types, and removing duplicates.
    - **Exploratory Data Analysis (EDA)**: Identifying patterns, trends, and correlations within the data using various visualization techniques.
    - **Feature Engineering**: Creating new features from the existing ones to improve model performance.
    - **Data Transformation**: Scaling, normalizing, and encoding categorical variables.
    - **Saving Processed Data**: Exporting the cleaned and transformed data for use in machine learning pipelines.

### Model_Building_and_Training.ipynb
- Focuses on building and training machine learning models and deep learning models to detect fraud cases.
- **Key Features**:
  - Uses libraries like `torch`, `sklearn`, and `mlflow`.
  - Loads cleaned datasets (fraud and credit card data) for training.
  - Implements various classifiers such as `LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier`.
  - Also implements several deep learning models like MLP, CNN, RNN and LSTM.
  - Splits data into training and testing sets to evaluate model performance.
  - Evaluates models with metrics like accuracy, precision, recall, F1-score, and log loss.
  - Tracks all models and metrics using MLflow for easy comparison.
  - Automatically logs parameters, metrics, and models using MLflow's autologging feature.

### Model_Explainability.ipynb
The notebook loads various pre-trained models (PyTorch and scikit-learn) for both credit scoring and fraud detection datasets. It also handles dataset preparation, removing irrelevant columns and performing train-test splits. The models are stored in a dictionary format for easy retrieval.

#### Explainability Analysis
This section leverages SHAP and LIME libraries for explainability:
1. **SHAP**: Applies SHAP explainability to traditional machine learning models and deep learning models separately. The `TreeExplainer` is used for tree-based models, while `KernelExplainer` is used for general-purpose explainability. For each model, SHAP summary plots highlight feature importance, with a specific focus on interpreting the model's predictions for fraud detection and credit risk assessment.
  
2. **LIME**: The notebook generates LIME explanations for both traditional machine learning and deep learning models, providing instance-specific feature contributions. Each LIME analysis focuses on a single sample to demonstrate how specific feature values contribute to the prediction.

### Explainability for PyTorch Models
Custom functions are implemented to compute SHAP values for PyTorch models, accommodating neural network architectures. Additionally, LIME explanations are generated for these models with a custom prediction function to handle binary classification probabilities.

## Installation

To run the notebook, you need to set up a Python environment with the necessary dependencies. The following steps outline how to do this:

1. Clone the repository:
    ```bash
    git clone https://github.com/yohannestayez/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
    cd your-repository
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch the Jupyter notebook:
    ```bash
    jupyter notebook Data_Analysis_and_Preprocessing.ipynb
    ```

## Dependencies

Make sure to install the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `PyTorch`
- `MLflow`
- `imblearn`
- `shap`
- `lime`
- `joblib`
- Any other dependencies used in the notebook can be found in the `requirements.txt` file.

## Usage

This notebook is ideal for data scientists and analysts looking to:
- Perform initial data exploration.
- Clean and preprocess data for machine learning tasks.
- Gain insights into the structure and properties of a dataset.

## Fraud and Credit Card Detection Model API

A Flask API for serving and monitoring fraud and credit card detection models, deployed in a Docker container. This API allows users to submit data for predictions on two models:
- **Fraud Detection Model** (Decision Tree model)
- **Credit Card Detection Model** (RNN model)

### Project Structure

```
model_api/
├── models/
│   ├── DecisionTree_Fraud.joblib        # Pre-trained fraud detection model
│   └── RNN_Credit.pt                    # Pre-trained credit card model
├── serve_model.py                       # Main Flask application
├── requirements.txt                     # Python dependencies
└── Dockerfile                           # Docker 
```

### Getting Started

#### Requirements

- Python 3.x
- Docker
- Flask and required dependencies (listed in `requirements.txt`)

#### Installation

1. Clone the repository:
   ```bash
      git clone https://github.com/yohannestayez/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
   cd fraud-creditcard-api/model_api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API locally:
   ```bash
   python serve_model.py
   ```

4. To test predictions, use `POST` requests to `/predict/fraud` or `/predict/creditcard` with JSON payloads.

### API Endpoints

- **`GET /`**: Home route; checks if the API is running.
- **`POST /predict/fraud`**: Sends JSON data to the fraud detection model.
- **`POST /predict/creditcard`**: Sends JSON data to the credit card detection model.


### Dockerization

1. Build the Docker image:
   ```bash
   docker build -t fraud-detection-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 5000:5000 fraud-detection-api
   ```

### Monitoring and Logging

- **Logging**: Logs incoming requests, prediction results, and errors. Logs are stored in `app.log`.
- **Future Extensions**: For advanced monitoring, integrate with monitoring solutions like Prometheus or Grafana.

## Dashboard

- This folder provides a web-based dashboard for analyzing fraud data and making predictions using machine learning models. It consists of a Flask API for model serving and a Dash application for interactive visualizations. more description in the folder.
- 📫 Feel free to checkout out the dashboard at <a href="https://detection-of-fraud-cases-for-e-commerce.onrender.com">https://detection-of-fraud-cases-for-e-commerce.onrender.com</a>

## Contributing

If you'd like to contribute, feel free to open a pull request or submit an issue. Contributions that enhance the analysis or add new preprocessing techniques are welcome!

