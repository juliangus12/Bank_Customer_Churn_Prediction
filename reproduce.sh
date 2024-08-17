#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if conda is installed
check_conda_installed() {
    if ! command -v conda &>/dev/null; then
        echo "Conda is not installed. Would you like to install it now? (yes/no)"
        read answer
        if [ "$answer" == "yes" ]; then
            echo "Installing Miniconda..."
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
            bash ~/miniconda.sh -b -p $HOME/miniconda
            export PATH="$HOME/miniconda/bin:$PATH"
            echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
            source ~/.bashrc
            conda init
        else
            echo "Conda installation skipped. Exiting script."
            exit 1
        fi
    else
        echo "Conda is already installed."
    fi
}

# Check if Conda is installed
check_conda_installed

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

# Step 5: Run the neural network script
echo "Running neural network script..."
python workflow/neural.py

echo "All steps completed successfully!"
