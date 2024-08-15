import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
full_data = pd.read_csv('data/Churn_Modelling.csv')

# display basic information about the datasets
print("train data info:")
print(train_data.info())
print("\ntest data info:")
print(test_data.info())
print("\nfull data info:")
print(full_data.info())

# display the first few rows of the train dataset
print("\ntrain data head:")
print(train_data.head())

# check for duplicated rows in the train dataset
print(f"\nnumber of duplicated rows in train data: {train_data.duplicated().sum()}")

# drop duplicated rows from the train dataset
train_data.drop_duplicates(inplace=True)
print(f"shape after removing duplicates: {train_data.shape}")

# check for missing values in the train dataset
print("\nmissing values in train data:")
print(train_data.isnull().sum())

# drop rows with missing values from the train dataset
train_data.dropna(inplace=True)
print(f"shape after removing rows with missing values: {train_data.shape}")

# display summary statistics after data cleaning
print("\nsummary statistics after data cleaning:")
print(train_data.describe())

# visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=train_data)
plt.title('distribution of churn (exited) in the training data')
plt.savefig('figures/churn_distribution.png')

# select only numeric columns for correlation matrix
numeric_columns = train_data.select_dtypes(include=[np.number])

# create a correlation matrix for the numeric columns
corr_matrix = numeric_columns.corr()

# visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('correlation matrix')
plt.savefig('figures/correlation_matrix.png')

# visualize the distribution of key numeric features
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(train_data['CreditScore'], kde=True)
plt.title('distribution of credit score')

plt.subplot(2, 2, 2)
sns.histplot(train_data['Age'], kde=True)
plt.title('distribution of age')

plt.subplot(2, 2, 3)
sns.histplot(train_data['Balance'], kde=True)
plt.title('distribution of balance')

plt.subplot(2, 2, 4)
sns.histplot(train_data['EstimatedSalary'], kde=True)
plt.title('distribution of estimated salary')

plt.tight_layout()
plt.savefig('figures/numeric_features_distribution.png')

# visualize the relationship between categorical features and the target variable
plt.figure(figsize=(12, 6))
sns.countplot(x='Gender', hue='Exited', data=train_data)
plt.title('churn distribution by gender')
plt.savefig('figures/gender_churn_distribution.png')

plt.figure(figsize=(12, 6))
sns.countplot(x='Geography', hue='Exited', data=train_data)
plt.title('churn distribution by geography')
plt.savefig('figures/geography_churn_distribution.png')

# use a pairplot to analyze relationships between key numeric features and the target variable
sns.pairplot(train_data, hue='Exited', vars=['CreditScore', 'Age', 'Balance', 'EstimatedSalary'])
plt.savefig('figures/pairplot_numeric_features.png')
