# Improved detection of fraud cases for e-commerce and bank transactions

## overview
This repository contains a Jupyter notebook focused on data analysis and preprocessing techniques. The notebook provides a comprehensive workflow for preparing datasets for further modeling and analysis, particularly for machine learning projects.

## Notebooks

### Data_Analysis_and_Preprocessing.ipynb
- This notebook demonstrates how to handle raw data, clean it, perform exploratory data analysis (EDA), and transform it into a format suitable for machine learning algorithms. The key steps covered in the notebook include:

1. **Loading the Data**: Reading raw data from various sources such as CSV, JSON, or Excel files.
2. **Data Cleaning**: Handling missing values, correcting data types, and removing duplicates.
3. **Exploratory Data Analysis (EDA)**: Identifying patterns, trends, and correlations within the data using various visualization techniques.
4. **Feature Engineering**: Creating new features from the existing ones to improve model performance.
5. **Data Transformation**: Scaling, normalizing, and encoding categorical variables.
6. **Saving Processed Data**: Exporting the cleaned and transformed data for use in machine learning pipelines.

### Model_Building_and_Training.ipynb
- Focuses on building and training machine learning models and deep learning models to detect fraud cases.
- **Key Features**:
  - Uses libraries like `torch`, `sklearn`, and `mlflow`.
  - Loads cleaned datasets (fraud and credit card data) for training.
  - Implements various classifiers such as `LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier`.
  - Splits data into training and testing sets to evaluate model performance.
  - Evaluates models with metrics like accuracy, precision, recall, F1-score, and log loss.
  - Tracks all models and metrics using MLflow for easy comparison.
  - Automatically logs parameters, metrics, and models using MLflow's autologging feature.


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
- Any other dependencies used in the notebook can be found in the `requirements.txt` file.

## Usage

This notebook is ideal for data scientists and analysts looking to:
- Perform initial data exploration.
- Clean and preprocess data for machine learning tasks.
- Gain insights into the structure and properties of a dataset.

## Contributing

If you'd like to contribute, feel free to open a pull request or submit an issue. Contributions that enhance the analysis or add new preprocessing techniques are welcome!

