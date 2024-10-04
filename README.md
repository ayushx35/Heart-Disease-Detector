# Heart Disease Prediction

## Overview

This project aims to predict the presence of heart disease in a patient using various machine learning algorithms. The prediction is based on a dataset containing various health metrics like age, cholesterol level, blood pressure, and more. This project implements and compares several classification models to identify the one that best predicts heart disease risk.

## Dataset

The dataset used for this project contains various medical attributes, which include:

Age
Sex
Chest pain type (4 values)
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results (values 0, 1, 2)
Maximum heart rate achieved
Exercise-induced angina
ST depression induced by exercise relative to rest
The slope of the peak exercise ST segment
Number of major vessels (0-3) colored by fluoroscopy
Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
Project Structure

data/: Contains the dataset files.
notebooks/: Jupyter Notebooks used for data exploration, feature engineering, model training, and evaluation.
src/: Contains the scripts for the machine learning models.
requirements.txt: Lists the dependencies and libraries required for running the project.
README.md: Project overview and instructions.
Dependencies

To run this project, install the necessary libraries by using the following command:

bash
Copy code
pip install -r requirements.txt
Main libraries include:

numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
How to Run


The following models have been implemented and compared for accuracy:

Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Random Forest Classifier
XGBoost
Each model is trained and evaluated based on performance metrics like accuracy, precision, recall, and F1 score. The final model is selected based on the best overall performance.

## Results

The project shows a comparison of various models, highlighting the one that achieves the highest prediction accuracy for heart disease. Additional evaluations are made on model interpretability, cross-validation scores, and error metrics.

## Conclusion

This project demonstrates the implementation of different machine learning algorithms for predicting heart disease. The selected model can help healthcare providers make better-informed decisions and potentially save lives by identifying high-risk patients earlier.

## Future Improvements

Implementation of additional advanced machine learning algorithms.
Optimization using hyperparameter tuning.
Deploying the model as a web application using Flask or Django.
