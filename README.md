# Salary Classification Using Job Descriptions

This repository shows how to deploy text analysis on job description and make classification on salary. The dataset is downloaded from Kaggle. 

## Objective
This project aims to predict whether a job posting corresponds to a high or low salary based on its job description. Using a dataset from Kaggle, a subset of 2500 data points was selected for analysis, split into training (80%) and test (20%) sets.The goal was to build a classification model that predicts salary ranges (high/low) from job descriptions. The model leverages Natural Language Processing (NLP) techniques and various machine learning classifiers to achieve the task.

## Methodology
Several machine learning models were tested to solve the classification problem:

- **Models Tested**:
  - Naïve Bayes
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Logistic Regression (LR)
  - Random Forest (RF)

- **Data Preprocessing**:
  - Text data was preprocessed using **NLTK** (Natural Language Toolkit), involving:
    - Tokenization
    - Stopword removal
  - **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to transform textual data into numerical format, emphasizing significant terms in job descriptions.

- **Hyperparameter Tuning**:
  - Each model underwent thorough hyperparameter tuning to maximize accuracy.

## Results
- The **SVM classifier** achieved the highest accuracy of **81%** on the test set.
- The model’s performance was assessed using a **confusion matrix** to evaluate true positives, false positives, true negatives, and false negatives.
- **High Salary Indicators**: The top 10 words associated with high salaries were:
  - ['senior', 'strategic', 'lead', 'social', 'director', 'global', 'partner', 'leadership', 'project', 'head']
  - These terms suggest that high-paying roles are often senior, leadership, and strategic positions.
  
- **Low Salary Indicators**: The top 10 words associated with low salaries were:
  - ['well', 'travel', 'assistant', 'manufacturing', 'wcf', 'hour', 'charity', 'database', 'applicant', 'customer']
  - These terms point to roles that tend to be lower-paying, such as administrative, customer service, or hourly-based jobs.

