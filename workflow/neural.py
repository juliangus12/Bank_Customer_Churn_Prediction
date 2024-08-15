import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# load the preprocessed data
X = pd.read_csv('data/processed_X.csv')
y = pd.read_csv('data/processed_y.csv')

# convert y to a 1d array if it's not already
y = y.values.ravel()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# drop the 'surname' column from x_train and x_test since it's not relevant for modeling
X_train = X_train.drop(columns=['Surname'])
X_test = X_test.drop(columns=['Surname'])

# scale the features to normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define the neural network model with l2 regularization to prevent overfitting
def create_model(learning_rate, beta1, beta2):
    model = Sequential()

    # input layer with batch normalization and dropout to reduce overfitting
    model.add(Dense(256, input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)))  # increased neurons with l2 regularization
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # increased dropout for regularization

    # hidden layer 1 with l2 regularization batch normalization and dropout
    model.add(Dense(128, kernel_regularizer=l2(0.001)))  # l2 regularization added
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    # hidden layer 2 with l2 regularization batch normalization and dropout
    model.add(Dense(64, kernel_regularizer=l2(0.001)))  # l2 regularization added
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    # output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # compile the model with adamw optimizer using dynamic learning rates and beta values
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# hyperparameter values to test including different learning rates and beta values
learning_rates = [0.001, 0.0005, 0.0001]
beta1_values = [0.9, 0.85]
beta2_values = [0.999, 0.995]

# best performance tracker to keep track of the best model's accuracy and parameters
best_accuracy = 0
best_params = {}

# callbacks for dynamic learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# fine-tuning loop to train the model with different combinations of hyperparameters
for lr in learning_rates:
    for b1 in beta1_values:
        for b2 in beta2_values:
            print(f"training with learning_rate={lr}, beta1={b1}, beta2={b2}")
            model = create_model(lr, b1, b2)
            
            # adjust class weights to handle class imbalance in the dataset
            class_weight = {0: 1.0, 1: 2.0}  # adjust these weights based on previous performance
            
            # train the model with the defined hyperparameters and callbacks
            history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1, 
                                class_weight=class_weight, callbacks=[reduce_lr, early_stopping])
            
            # evaluate the model on the test data
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"validation accuracy: {accuracy:.4f}")
            
            # update the best accuracy and parameters if the current model performs better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'beta1': b1, 'beta2': b2}

# print the best accuracy and corresponding parameters
print(f"best accuracy: {best_accuracy:.4f} with parameters: {best_params}")