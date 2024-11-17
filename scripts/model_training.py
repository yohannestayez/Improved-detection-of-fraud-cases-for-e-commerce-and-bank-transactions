import torch
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle



def augment_and_merge_data(original_data, target_column, num_samples=None, random_state=42):
    """
    Augment data using SMOTE and merge synthetic data with the original dataset.
    
    Args:
        original_data (pd.DataFrame): Original dataset.
        target_column (str): Name of the target column in the dataset.
        num_samples (int, optional): Number of synthetic samples to generate.
                                     Defaults to balancing the minority class with SMOTE.
        random_state (int, optional): Random state for reproducibility. Default is 42.
        
    Returns:
        pd.DataFrame: Augmented and shuffled dataset containing both original and synthetic data.
    """
    # Separate features and target
    X = original_data.drop(columns=[target_column])
    y = original_data[target_column]

    # Instantiate SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    
    # Apply SMOTE to generate synthetic data
    print("Applying SMOTE to generate synthetic samples...")
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Combine features and target back into a DataFrame
    synthetic_data = pd.DataFrame(X_resampled, columns=X.columns)
    synthetic_data[target_column] = y_resampled
    
    # Ensure no duplicate rows are added
    synthetic_data = synthetic_data[~synthetic_data.isin(original_data)].dropna(how='all')
    
    # Limit synthetic data size if num_samples is specified
    if num_samples:
        synthetic_data = synthetic_data.sample(n=num_samples, random_state=random_state)
    
    # Merge with original data
    print("Merging synthetic and original data...")
    augmented_data = pd.concat([original_data, synthetic_data], ignore_index=True)
    
    # Shuffle the data
    print("Shuffling merged dataset...")
    augmented_data = shuffle(augmented_data, random_state=random_state)
    
    print("Original data shape:", original_data.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    print("Augmented data shape:", augmented_data.shape)
    # Return the augmented dataset
    print("Data augmentation and merging completed.")
    return augmented_data


# Scikit-learn classifiers
def train_sklearn_model(model, X_train, y_train):
    model.fit(X_train, y_train)

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# Training loop with MLflow tracking
def train_model(model, train_loader, optimizer, criterion, num_epochs=20, patience=3, model_name="model"):
    early_stopper = EarlyStopping(patience=patience)
    model.train()
    
    with mlflow.start_run(run_name=model_name):
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # Clear gradients
                y_pred = model(X_batch).squeeze()  # Forward pass
                loss = criterion(y_pred, y_batch)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Log loss for each epoch
            mlflow.log_metric('loss', avg_loss, step=epoch)

            # Early stopping
            early_stopper(avg_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered!")
                break

    # Model evaluation
# Updated PyTorch model evaluation function
def evaluate_model(model, test_loader):
    model.eval()  # Set to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            preds = (y_pred > 0.5).float()  # Convert probabilities to 0/1
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy:.4f}')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def evaluate_sklearn_model(model, X_test, y_test, model_name):
    with mlflow.start_run(run_name=model_name):
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

