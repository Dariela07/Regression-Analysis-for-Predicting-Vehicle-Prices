import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import seaborn as sns

df = pd.read_csv("clean_data_le.csv")
print(df.shape)

features = ['engine-size', 'horsepower', 'curb-weight', 'highway-L/100km']
plt.figure(figsize=(12, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.4)
for i, ft in enumerate(features):
    plt.subplot(4, 3, 1 + i * 2)
    sns.regplot(x=ft, y='price', data=df)
    plt.subplot(4, 3, 2 + i * 2)
    sns.residplot(x=ft, y='price', data=df)
# categorical:
features_ca = ['engine-location', 'drive-wheels']
for i, ft in enumerate(features_ca):
    plt.subplot(4, 3, i + 9)
    sns.boxplot(x=ft, y="price", data=df)
plt.suptitle("Potential features for multiple linear regression")
# plt.show()
# plt.close()

# Capture non-linear residuals with polynomial features for 'horsepower', 'curb-weight'
x = df[['engine-size', 'horsepower', 'curb-weight', 'engine-location', 'drive-wheels']]
y = df[['price']]
pr = PolynomialFeatures(degree=5)
x_horsepower_poly = pd.DataFrame(pr.fit_transform(x[['horsepower']]))
# x_curbweight_poly = pd.DataFrame(pr.fit_transform(x[['curb-weight']]))
x_horsepower_poly.columns = ['hp_p0', 'hp_p1', 'hp_p2', 'hp_p3', 'hp_p4', 'hp_p5']

x_new = x.drop(columns=['horsepower'])
x_data = pd.concat([x_new, x_horsepower_poly], axis=1)
print(x_data)
print(x_data.columns)
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.3, random_state=1)

# print(x_train_pf_horsepower.head())
lm_pf = LinearRegression()
lm_pf.fit(x_train, y_train)

print('Training data R2', lm_pf.score(x_train, y_train))
print('Test data R2', lm_pf.score(x_test, y_test))

# Training data R2 0.8592850017984738
# Test data R2 0.7944890026304663

lm2 = LinearRegression()
lm2.fit(x_data, y)
Rcross = cross_val_score(lm2, x_data, y, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())
# print(cross_val_predict(lm2, x_data, y, cv=4))

# [0.76997706 0.81713241 0.61814284 0.37969634]
# The mean of the folds are 0.6462371602995733 and the standard deviation is 0.17055061405291533

# Not much improvement adding polynomial feature

# Ridge Regression controls coefficients of magnitude by introducing alpha, prevent over fitting
