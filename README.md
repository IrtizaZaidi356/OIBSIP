# OIBSIP
# **Data Science | Remote Internship | Oasis InfoByte**
## Perfom to Task No# 01: IRIS Flower Classification
  - Iris flower has three species, setosa, versicolor, and virginica, which differs according to their measurements. Now assume that you have the measurements of the iris flowers according to their species and here your task is to train a machine learning model that can learn from the measurements of the iris species and classify them.
  - Although the Scikit-learn library provides a dataset for iris flower classification

---

## **Iris Flower Classification Project:**

In this Iris Flower Classification project, we aimed to analyze and predict the species of Iris flowers based on their sepal and petal characteristics. The project followed a structured approach, encompassing data exploration, visualization, preprocessing, model training, and evaluation.

#### **Data Overview:**
  - We began by importing necessary libraries and loading the Iris Flower dataset. The dataset comprised 150 rows and 6 columns, containing features such as sepal length, sepal width, petal length, petal width, and the species of the Iris flowers.

#### **Data Cleaning and Exploration:**
   - We checked for missing values, duplicate rows, and visualized the distribution of data using various plots. Outliers were detected using boxplots, and the count of each species was visualized through bar charts. Data distributions and relationships were explored through histograms and scatter plots.

#### **Data Preprocessing:**
  - The dataset was split into features (X) and the target variable (y). Subsequently, the data was divided into training and testing sets using the train_test_split function. Standardization was applied using the StandardScaler to ensure consistent scaling across features.

#### **Model Training and Evaluation:**
  - We chose the K-Nearest Neighbors (KNN) algorithm for classification and trained the model using the training set. Predictions were made on the test set, and the model's performance was evaluated using a confusion matrix, classification report, and accuracy score.

#### **Results and Prediction:**
  - The model achieved an accuracy of [insert accuracy]% on the test set. To showcase the model's usability, a user input for prediction was incorporated. The input data was appropriately scaled using the fitted scaler, and the model predicted the Iris species based on the provided features.

In **Conclusion**, this project demonstrated a comprehensive analysis of Iris flower data, including visualization, preprocessing, and model building. The trained KNN model can effectively predict the species of Iris flowers based on their characteristics.
