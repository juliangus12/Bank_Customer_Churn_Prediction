#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Set up the environment
echo "Setting up the conda environment..."
conda env create -f environment.yml

# Activate the conda environment
echo "Activating the environment..."
source activate churn-prediction

# Step 2: Preprocess the data
echo "Running data preprocessing..."
python workflow/preProcessing.py

# Step 3: Perform Exploratory Data Analysis (EDA)
echo "Running Exploratory Data Analysis (EDA)..."
python workflow/EDA.py

# Step 4: Train and evaluate models
echo "Training and evaluating models..."
python workflow/model_training.py

echo "All steps completed successfully!"
