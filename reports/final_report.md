# Final Report: Predicting Bank Customer Churn Using Regression and Neural Networks

## Introduction

Banks are very concerned about customer turnover because it is usually less expensive to keep current clients than to find new ones. Banks can use targeted retention methods, including tailored marketing campaigns or improved customer service, by identifying which customers are most likely to leave. This project's goal was to use a variety of machine learning models, such as neural networks and regression approaches, to forecast if a client would leave. We aimed to determine the most successful method for forecasting customer attrition and obtain understanding of the fundamental elements that lead to attrition by contrasting the performance of different models.


## Motivation

In the banking sector, where there is intense competition and client retention is crucial, predicting customer churn is essential. Banks need to be able to pinpoint the causes of client attrition and utilize predictive models to act before consumers depart. In order to ascertain which machine learning strategies produce the best results and offer the most insightful information on consumer behavior, this research investigated a variety of approaches. We also looked at how several features, like activity levels, product usage, and customer age, affected the likelihood of churn. Our objective was to offer useful information that the bank might use to guide its client retention tactics.


## Data Description
The dataset utilized for this project is sourced from ABC Multistate Bank and comprises a range of financial and demographic data on the customers as well as details about their activities. Credit score, location, gender, age, tenure, balance, quantity of items, credit card ownership, active membership, and expected salary are some of the important characteristics that are incorporated into the models. The binary target variable, churn, represents whether a customer has stayed with the bank (0) or left (1). Building models that correctly predict this target variable from the available input features was our objective.

The dataset used in this project comes from ABC Multistate Bank. It includes customer demographic information, financial data, and activity status, with a binary target variable indicating whether the customer has churned or not. The key features used in the models are:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

The goal is to predict the target variable, `churn`

## Methods

We employed several machine learning models, including:
- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **Neural Networks (Adam Optimizer)**

Each model was tuned using cross-validation and hyperparameter optimization techniques. For the Gradient Boosting model, we used `RandomizedSearchCV` to identify the optimal hyperparameters, while the neural network was optimized using the Adam optimizer with tuned learning rates and beta values.

### Key Insights from the Data

Our model building was informed by numerous significant associations that our exploratory data analysis uncovered. The age group of consumers exhibited a robust positive link with bank churn, indicating a higher likelihood of bank attrition. This research suggests that clients may turn to other financial organizations for services if their banking demands change as they get older. However, there were significant negative associations found between turnover and a customer's amount of items as well as whether or not they are an active member. This suggests that clients who have a higher level of engagement with the bank—either by actively participating or by having a variety of products—have lower attrition rates. These realizations were crucial in helping us choose the features for our models.

## Results

### Model Performance Comparison:

- **Linear Regression**: Accuracy: 0.8602
- **Lasso Regression**: Accuracy: 0.8182
- **Ridge Regression**: Accuracy: 0.8602
- **Logistic Regression**: Accuracy: 0.8244
- **Random Forest**: Accuracy: 0.8878
- **Gradient Boosting**: Accuracy: 0.8960
- **Neural Networks (Adam)**: Best accuracy: 0.8751 with parameters: {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.995}

The Gradient Boosting model performed the best overall, achieving an accuracy of 0.8960.

### Tuning Results

For the Gradient Boosting model, hyperparameter tuning yielded the following performance:
- **Accuracy**: 0.8927
- **Precision for Churned Customers**: 0.80
- **Recall for Churned Customers**: 0.65
- **F1-Score for Churned Customers**: 0.72

This performance demonstrates that the Gradient Boosting model was able to capture the underlying patterns in the data effectively. Despite the Neural Network's promising results, Gradient Boosting showed better overall performance, particularly in precision and recall for churned customers.

## Learning Outcomes

We learned a lot about applying different machine learning approaches to real-world problems throughout this research. Learning how to use the Adam optimizer to optimize a neural network was one of the most important lessons. We were able to greatly enhance the network's performance by adjusting the learning rate and beta values. We also investigated tweaking hyperparameters for Gradient Boosting models, effectively searching the hyperparameter space with RandomizedSearchCV. This procedure made clear how crucial adjusting is to raising the resilience and accuracy of the model. In addition to honing our technical abilities, this project taught us the ins and outs of organizing and planning a data science competition project, and we're excited to take part in additional Kaggle competitions in the future.

## Author Contributions

- **Daniel Duan**: Focused on hyperparameter tuning for the Gradient Boosting model, conducting experiments to find the optimal set of hyperparameters and improve model performance.

- **Julian Gutierrez**: Led the development and optimization of the neural network model, with a focus on fine-tuning the Adam optimizer. Julian also performed exploratory data analysis and feature engineering for the project.

## Conclusion

This research effectively illustrated how ABC Multistate Bank may use machine learning approaches to forecast customer attrition. We were able to forecast customer attrition with a high degree of accuracy by comparing several regression models and a neural network. Banks can create focused initiatives to retain at-risk consumers with the assistance of the insights obtained from this investigation, especially about the significance of customer engagement and product utilization. In the end, this project demonstrates the effectiveness of machine learning in resolving challenging business issues and lays the groundwork for future research into predictive modeling.
