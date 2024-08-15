import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the preprocessed data
X = pd.read_csv('data/processed_X.csv')
y = pd.read_csv('data/processed_y.csv')

# Convert y to a 1D array if it's not already
y = y.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Drop the 'Surname' column from X_train and X_test
X_train = X_train.drop(columns=['Surname'])
X_test = X_test.drop(columns=['Surname'])

# Check that there are no non-numeric columns left
non_numeric_columns = X_train.select_dtypes(include=['object']).columns
print("Non-numeric columns in X_train after dropping 'Surname':", non_numeric_columns)

# Initialize models
linear_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()
logistic_model = LogisticRegression(max_iter=1000)
random_forest_model = RandomForestClassifier(random_state=42)
gradient_boost_model = GradientBoostingClassifier(random_state=42)

# Train the models
linear_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
gradient_boost_model.fit(X_train, y_train)

# Output the training completion
print("Models have been trained.")

# Create a dictionary to store the model predictions and their performance metrics
model_performance = {}

# List of models to evaluate
models = {
    'Linear Regression': linear_model,
    'Lasso Regression (L1)': lasso_model,
    'Ridge Regression (L2)': ridge_model,
    'Logistic Regression': logistic_model,
    'Random Forest': random_forest_model,
    'Gradient Boosting': gradient_boost_model
}

# Adjusted threshold for regression models
threshold = 0.5

# Evaluate each model
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    
    if model_name in ['Linear Regression', 'Lasso Regression (L1)', 'Ridge Regression (L2)']:
        y_pred = (y_pred > threshold).astype(int)  # Convert to binary based on threshold
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, zero_division=1)
    
    # Store the results in the dictionary
    model_performance[model_name] = {
        'Accuracy': accuracy,
        'Confusion Matrix': confusion,
        'Classification Report': classification_rep
    }
    
    # Print the results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", classification_rep)
    print("-" * 50)
