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

---

## **Perform Task 04: Email Spam detection with Machine Learning:**
  - We have all been the recipient of spam emails before, Spam mail, or junk mail is a type of email that is sent to a  massive number of users at one time, frequently containing cryptic messages, scams, or most dangerously, phishing content.

---

### **Project No# 04: Email Spam Detection with Machine Learning:**

#### **Introduction:**
  - The Email Spam Detection project aims to build a machine learning model that can effectively classify emails as spam or ham (non-spam). The dataset used for this project contains information about email messages, including labels indicating whether they are spam or ham.

#### **Step 1: Importing Libraries and Setup:**
  - In this step, essential libraries such as NumPy, Pandas, Seaborn, and Scikit-Learn are imported. Warnings are filtered to enhance the clarity of the presentation. The dataset is loaded, and a brief exploration is performed to understand its structure.

#### **Step 2: Data Exploration:**
   - This step involves exploring the dataset, checking for missing values, and summarizing key statistics. Unnecessary columns are dropped for simplicity, and the remaining columns are renamed for clarity. The distribution of spam and ham emails is visualized using a bar chart, and the message length distribution is examined through a histogram.

#### **Step 3: Data Visualization:**
  - Missing values are visualized using a heatmap. The distribution of spam and ham emails is further explored with a pair plot, providing insights into the relationships between different features.

#### **Step 4: Data Preprocessing:**
  - Labels are converted into numerical values (spam: 1, ham: 0) to prepare the data for model training.

#### **Step 5-6: Feature Extraction and Splitting Data:**
  - Text data is converted into a bag-of-words model using CountVectorizer. The data is then split into training and testing sets.

#### **Step 7: Model Training:**
  - A Multinomial Naive Bayes classifier is chosen for email spam detection and trained on the dataset.

#### **Step 8: Model Evaluation:**
  - The model is evaluated using the testing set, and metrics such as accuracy, confusion matrix, and classification report are displayed to assess its performance.

#### **Step 9: User Input and Prediction:**
  - The user is prompted to input an email text, and the trained model predicts whether the input email is spam or ham.

#### **Conclusion:**
  - This project demonstrates the entire process of building an email spam detection system using machine learning. Python libraries like NumPy, Pandas, and Scikit-Learn are instrumental in data manipulation, exploration, and model training. The Multinomial Naive Bayes algorithm proves effective for this classification task. The user interface allows for practical use of the model, showcasing the real-world applicability of machine learning in email filtering and cybersecurity.

---

## **Perform Task 05: Sales Prediction using Python:**
 - `Sales prediction` means predicting how much of a product people will buy based on factors such as the amount you spend to advertise your product, the segment of people you advertise for, or the platform you are `advertising` on about your `product`.
 - Typically, a product and service – based business always need their `Data Scientist` to predict their future sales with every step they take to manipulate the cost of advertising their product. So let’s start the `task` of `Sales Prediction` with `Machine Learning` using `Python`.

---
### **Project No# 05: Sales Prediction using Python**

#### **Introduction:**
  - Sales prediction is crucial for businesses as it helps in planning and strategizing for future growth. In this project, we used Python and various libraries to perform Exploratory Data Analysis (EDA) and build a machine learning model for sales prediction.

#### **Data Exploration and Visualization:**
  - We started by importing necessary libraries and loading the dataset, which included features like TV, Radio, Newspaper ad expenses, and Sales. After handling missing values and creating a new feature 'Total_Ad_Expenses', we visualized the data using various charts.

    - **Heatmaps and Correlation Matrix:** Explored the correlation between features.
    - **Box Chart and Histogram:** Visualized the distribution of Sales.
    - **Pair Plot:** Examined relationships between different features.
    - **Scatter Plots:** Explored the impact of TV, Radio, and Newspaper expenses on Sales.
    - **Pie Charts:** Analyzed the distribution of ad expenses with respect to the total.

#### **Data Preprocessing and Feature Engineering:**
  - We handled missing values, dropped unnecessary columns, and created a new feature for total ad expenses. The data was standardized for machine learning.

#### **Machine Learning Model:**
  - We used a Linear Regression model to predict sales based on TV, Radio, and Newspaper ad expenses. The model was trained, and predictions were made on the test set. Mean Squared Error was used for evaluation.

#### **User Input and Prediction:**
  - The user could input values for TV, Radio, and Newspaper ad expenses, and the model predicted the sales. This feature allows businesses to get quick predictions based on their specific ad spend.

#### **Conclusion:**
  - In conclusion, this project demonstrated the end-to-end process of sales prediction using Python. EDA helped in understanding the data, and the machine learning model provided valuable insights for decision-making. The user input feature adds practicality, allowing businesses to make real-time predictions for their advertising strategies. Sales prediction is a powerful tool for optimizing advertising budgets and maximizing revenue.
