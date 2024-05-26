import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict
import seaborn as sns

df = pd.read_csv("clear_data.csv")
# print(df.head())
# print(df.dtypes)
# print(df.columns)

# Develop a Multiple Linear Regression Model first: using these variables as the predictor variables.
# Select features with strong linear correlation with price.
le = preprocessing.LabelEncoder()
le.fit(df['engine-location'])
df['engine-location'] = le.transform(df['engine-location'])
print(le.classes_)
le.fit(df['drive-wheels'])
df['drive-wheels'] = le.transform(df['drive-wheels'])
print(le.classes_)

df.to_csv("clear_data_le.csv")

lm = LinearRegression()
predictors1 = df[['engine-size', 'engine-location', 'horsepower', 'curb-weight', 'drive-wheels']]
print(predictors1.head())
Y = df[['price']]

# Train and test data
x_train, x_test, y_train, y_test = train_test_split(predictors1, Y, test_size=0.2, random_state=3)

lm.fit(x_train, y_train)
print(f'Intercept equals: {lm.intercept_}, coefficients are: {lm.coef_}')

Y_hat = lm.predict(x_train)
# print(Y_hat)
y_test_hat = lm.predict(x_test)
# Check the R square for the training and test datasets
r2_test = lm.score(x_test, y_test)
r2_train = lm.score(x_train, y_train)
print("The R squared for test data is: ", r2_test)
print("The R squared for training data is: ", r2_train)

# The R squared for test data is:  0.872992009168348
# The R squared for training data is:  0.8268002020059249

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(wspace=0.3)
ax1 = sns.kdeplot(y_train, color='red', label="Actual Value")  # kernel density estimate
sns.kdeplot(Y_hat, color="y", label="Fitted Values", ax=ax1)
plt.title('Actual vs Fitted Values for Price in Training Data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.subplot(1,2,2)
ax1 = sns.kdeplot(y_test, color="red", label="Actual Value")  # kernel density estimate
sns.kdeplot(y_test_hat, color="b", label="Fitted Values", ax=ax1)
plt.title('Actual vs Fitted Values for Price in Test Data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()

# Cross Validation: out-of-sample evaluation metrics (average results as the estimate of out of sample error) Each
# observation is used for both training and testing Close to true generalization error but precision is poor,
# if more data training, less test; vice versa, poor performance but good precision

lm2 = LinearRegression()
lm2.fit(predictors1, Y)
Rcross = cross_val_score(lm2, predictors1, Y, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# [0.81884533 0.8091561  0.55644969 0.39258623]
# The mean of the folds are 0.6442593370748241 and the standard deviation is 0.1793886161099116
# print(cross_val_predict(lm, predictors1, Y, cv=4))