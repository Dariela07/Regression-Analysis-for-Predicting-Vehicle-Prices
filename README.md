# Regression-Analysis-for-Predicting-Vehicle-Prices
## 1. Dataset
This project uses the automobile dataset available at: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

The columns in the CSV dataset are:
"symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
"drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price".

## 2. Objective
The objective is to predict vehicle prices using machine learning regression models.

## 3. Methodologies

The analysis is performed using the pandas library in Python.

The analysis is divided into five phases:

<b> Data Preparation and Wrangling </b>: This includes reading data from a URL, replacing non-textual characters, handling missing values (dropping rows, mean or mode replacement), correcting data formats, data standardization, data normalization, binning, and converting categorical variables into dummy variables.

<b> Exploratory Data Analysis</b>: The aim is to identify the main characteristics that have the most impact on car prices. Continuous numerical variables (of type "int64" or "float64") are visualized using scatterplots with fitted lines. Categorical variables (of type "object" or "int64") are visualized using boxplots. Next, I performed "Descriptive Statistical Analysis" using the describe() function, value_counts(), pandas groupby method, pivot charts, heatmaps, Pearson correlations of variables with prices, Analysis of Variance (ANOVA), and non-linear relationship analysis through visualization.

<b>Multiple Linear Regression and Evaluation</b>: Multiple linear regression models are experimented with, depending on the features selected in the previous step. The R-squared for the test data is 0.873, and for the training data is 0.827.

<b>Polynomial Features</b>: Polynomial features for 'horsepower' and 'curb-weight' are added to capture non-linear residuals. However, there is not much improvement by adding polynomial features.

<b>Ridge Regression</b>: Moreover, Ridge regression techniques are experimented with.
 
