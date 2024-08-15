import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures

# load the dataset from the data folder
train_data = pd.read_csv('data/train.csv')

# display the first few rows of the dataset to check the data structure
print(train_data.head())

# separate numeric and non-numeric columns
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns

# fill missing values for numeric columns with the median value
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())

# fill missing values for non-numeric columns with the most frequent value
train_data[non_numeric_cols] = train_data[non_numeric_cols].fillna(train_data[non_numeric_cols].mode().iloc[0])

# verify that no missing values remain in the dataset
print("missing values after filling:\n", train_data.isnull().sum())

# apply label encoding to the gender column to convert it to numeric values
label_encoder = LabelEncoder()
train_data['Gender'] = label_encoder.fit_transform(train_data['Gender'])

# apply one-hot encoding to the geography column and drop the first category to avoid multicollinearity
train_data = pd.get_dummies(train_data, columns=['Geography'], drop_first=True)

# display the first few rows after encoding to check the transformations
print(train_data.head())

# create a new feature by multiplying balance and isactivemember to capture interaction effects
train_data['Balance_Active'] = train_data['Balance'] * train_data['IsActiveMember']

# generate polynomial features for creditscore age and balance to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(train_data[['CreditScore', 'Age', 'Balance']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['CreditScore', 'Age', 'Balance']))

# combine the polynomial features with the original data
train_data = pd.concat([train_data, poly_df], axis=1)

# display the first few rows after feature engineering to verify the changes
print(train_data.head())

# select numerical columns for scaling to ensure all features are on the same scale
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# standardize the numerical columns to normalize the data
scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])

# display the first few rows after scaling to confirm the standardization
print(train_data.head())

# separate the features and the target variable
X = train_data.drop('Exited', axis=1)
y = train_data['Exited']

# display the shapes of x and y to ensure they are correct
print("shape of x:", X.shape)
print("shape of y:", y.shape)

# save the preprocessed data to csv files for future use
X.to_csv('data/processed_X.csv', index=False)
y.to_csv('data/processed_y.csv', index=False)