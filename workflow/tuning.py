# this code was entirely generated by GPT-4o

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import uniform, randint

# load the preprocessed data
X = pd.read_csv('data/processed_X.csv')
y = pd.read_csv('data/processed_y.csv')

# convert y to a 1d array if it's not already
y = y.values.ravel()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# drop the 'surname' column from x_train and x_test as it is not relevant for modeling
X_train = X_train.drop(columns=['Surname'])
X_test = X_test.drop(columns=['Surname'])

# baseline gradient boosting model
# train a baseline gradient boosting model
baseline_gb = GradientBoostingClassifier(random_state=42)
baseline_gb.fit(X_train, y_train)

# evaluate the baseline model on the test data
y_pred_baseline = baseline_gb.predict(X_test)
print("baseline gradient boosting model performance:")
print(f"accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print("confusion matrix:\n", confusion_matrix(y_test, y_pred_baseline))
print("classification report:\n", classification_report(y_test, y_pred_baseline))

# hyperparameter tuning with randomizedsearchcv
# define the hyperparameters and their respective distributions to sample from
param_dist = {
    'n_estimators': randint(100, 300),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 6),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'subsample': uniform(0.8, 0.2)
}

# initialize randomizedsearchcv with parallel processing to optimize the gradient boosting model
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # number of different combinations to try
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # use all available cpu cores
    verbose=2,
    random_state=42
)

# perform randomizedsearchcv to find the best parameters for the gradient boosting model
random_search.fit(X_train, y_train)

# output the best parameters found by randomizedsearchcv
print("best parameters found by randomizedsearchcv:")
print(random_search.best_params_)

# evaluating the tuned gradient boosting model
# evaluate the best model on the test data using the best parameters found by randomizedsearchcv
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# calculate accuracy confusion matrix and classification report for the tuned model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# print the performance metrics for the tuned model
print(f"test accuracy: {accuracy:.4f}")
print("confusion matrix:\n", confusion)
print("classification report:\n", classification_rep)
