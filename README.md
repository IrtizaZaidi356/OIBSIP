# OIBSIP
# **Data Science | Remote Internship | Oasis InfoByte**

## **Perfom to Task No# 01: IRIS Flower Classification**
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

---

## **Perfom to Task No# 02: Unemployment Analysis with Python**
  - Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour face. We have seen a sharp increase in the unemployment rate during Covid – 19, so analyzing the unemployment rate can be a good data science project.

---

### **Task No# 02: Unemployment Analysis with Python:**

#### **Introduction:**
  - The Unemployment Analysis with Python project aimed to explore and analyze the trends in unemployment rates during the COVID-19 pandemic in India. Leveraging various Python libraries such as NumPy, Pandas, Seaborn, and Matplotlib, the project followed a systematic approach to data exploration, visualization, and machine learning modeling.

#### **Data Exploration and Cleaning:**
  - The dataset, sourced from 'Unemployment in India,' underwent thorough exploration and cleaning. Initial assessments of data shape, types, and missing values were conducted. Rows with null values were dropped to ensure data integrity. The 'Date' column was converted to a datetime format, and a new 'Year' column was created for further analysis.

#### **Visualization:**
  - Multiple visualizations were generated to provide insights into unemployment trends. Bar charts showcased the unemployment rate variation across different regions, while box charts illustrated the distribution of labor participation rates. The project also included the presentation of multiple charts comparing unemployment rates and employed populations by region, providing a comprehensive visual overview of the dataset.

  - A pair plot visually represented relationships between estimated unemployment rates, employed populations, labor participation rates, and years across different regions. Histograms and scatter plots were utilized to delve deeper into the distribution of unemployment rates and explore potential correlations between employed and unemployed populations.

#### **Machine Learning Model:**
  - The project implemented a machine learning model using linear regression to predict unemployment rates based on features such as estimated employment, labor participation rates, and years. The model achieved accurate predictions, as evidenced by a low Mean Squared Error (MSE). Users can interactively input values for estimated employment, labor participation rates, and years to receive predictions from the trained model.

#### **Conclusion:**
  - The Unemployment Analysis with Python project successfully demonstrated the application of data science techniques in exploring and understanding unemployment trends. Through systematic data exploration, insightful visualizations, and the implementation of a predictive model, the project provides a comprehensive overview of the dataset, aiding in the understanding and analysis of unemployment patterns during a critical period in India.

---

## **Perform Task No# 03: Car Price Prediction with Machine Learning:**

 - The price of a car depends on a lot of factors like the goodwill of the brand of the car, features of the car, horsepower and the mileage it gives and many more. Car price prediction is one of the major research areas in machine learning. So if you want to learn how to train a car price prediction model.
 
---

### **Task No# 03: Car Price Prediction with Machine Learning:**

#### **Introduction:**
  - The Car Price Prediction project is an exploration of machine learning techniques to predict the selling price of cars based on various features. This project utilizes Python libraries such as NumPy, Pandas, Seaborn, Matplotlib, Plotly, and scikit-learn for data analysis, visualization, and model building.

#### **Step 1: Importing Libraries and Handling Warnings:**
  - The project begins by importing essential libraries and setting up the environment. Warnings are suppressed for cleaner output. Additionally, the code adjusts the figure size for better visualizations.

#### **Step 2: Data Loading, Exploration, and Visualization:**
  - The dataset, containing information on car features, is loaded and explored. Initial data analysis includes checking the structure, summary statistics, and distribution of various features. Visualization techniques such as bar charts, box charts, pair plots, histograms, scatter plots, and bubble charts are employed to gain insights into the relationships between different variables.

#### **Outlier Removal:**
  - Outliers are identified and removed using the Interquartile Range (IQR) method to enhance the accuracy of the machine learning model.

#### **Step 3: Machine Learning Model:**
  - The dataset is prepared for the machine learning model by selecting relevant features and the target variable. It is then split into training and testing sets. A Linear Regression model is employed for predicting car selling prices. The model is trained, predictions are made, and the Mean Squared Error is calculated to evaluate its performance.

#### **Step 4: User Input and Prediction:**
  - To enhance user interaction, the project allows users to input car details, including manufacturing year, present price, driven kilometers, and the number of previous owners. The Linear Regression model then predicts the selling price based on this input.

#### **Conclusion:**
  - In conclusion, the Car Price Prediction project demonstrates a comprehensive workflow in machine learning, encompassing data exploration, visualization, outlier handling, model training, and user interaction. The insights gained from visualizations and the predictive capabilities of the model make this project valuable for anyone interested in the field of car price prediction using machine learning.

