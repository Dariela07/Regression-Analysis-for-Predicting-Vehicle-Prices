import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import seaborn as sns

df = pd.read_csv("clean_data_le.csv")
print(df.shape)
x = df[['engine-size', 'horsepower', 'curb-weight', 'engine-location', 'drive-wheels']]
y = df[['price']]
pr = PolynomialFeatures(degree=2)
x_pf = pr.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_pf, y, test_size=0.3, random_state=1)

# parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000], 'normalize': [True, False]}]
parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
print('1')
Grid = GridSearchCV(Ridge(), parameters, cv=4)
Grid.fit(x_train, y_train)

BestRR = Grid.best_estimator_
print(BestRR)
print("train R square", BestRR.score(x_train, y_train))
print("test R square", BestRR.score(x_test, y_test))
BestRR.score(x_test, y_test)
y_train_predict = BestRR.predict(x_train)
y_test_predict = BestRR.predict(x_test)
# Ridge(alpha=10000)
# train R square 0.8982699461063111
# test R square 0.8101615693702044


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(wspace=0.3)
ax1 = sns.kdeplot(y_train, color='blue', label="Actual Value")  # kernel density estimate
sns.kdeplot(y_train_predict, color="pink", label="Fitted Values", ax=ax1)
plt.title('Actual vs Fitted Values for Price in Training Data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.subplot(1,2,2)
ax1 = sns.kdeplot(y_test, color="red", label="Actual Value")  # kernel density estimate
sns.kdeplot(y_test_predict, color="b", label="Fitted Values", ax=ax1)
plt.title('Actual vs Fitted Values for Price in Test Data')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.legend()
plt.show()
