import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy import stats

# Explore "What are the main characteristics which have the most impact on the car price?".

df = pd.read_csv("clean_data.csv")
print(df.head())
print(df.dtypes)  # what type of variable you are dealing with

# Analyzing Individual Feature Patterns Using Visualization
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

# Continuous numerical variables:
# of type "int64" or "float64", good to visualize using scatterplots with fitted lines. Understand (linear)
# relationship between an individual variable and the price, use "regplot" (scatterplot, fitted regression line)
plt.figure(figsize=(15, 10))
# plt.tight_layout() # pad=5
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.subplot(2, 2, 1)
sns.regplot(x='engine-size', y='price', data=df).set(
    title=f'Correlation equals: {df[["engine-size", "price"]].corr().iloc[0, 1]:.2f}')
plt.subplot(2, 2, 2)
sns.regplot(x="highway-L/100km", y="price", data=df).set(
    title=f'Correlation equals: {df[["highway-L/100km", "price"]].corr().iloc[0, 1]:.2f}')
plt.subplot(2, 2, 3)
sns.regplot(x="peak-rpm", y="price", data=df).set(
    title=f'Correlation equals: {df[["peak-rpm", "price"]].corr().iloc[0, 1]:.2f}')
plt.subplot(2, 2, 4)
sns.regplot(x='stroke', y='price', data=df).set(
    title=f'Correlation equals: {df[["stroke", "price"]].corr().iloc[0, 1]:.2f}')
plt.suptitle("Continuous numerical variables")
# plt.show()

# Categorical Variables:
# type "object" or "int64". A good way to visualize categorical variables is by using boxplots.
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.subplot(2, 2, 1)
sns.boxplot(x="body-style", y="price", data=df)
plt.subplot(2, 2, 2)
sns.boxplot(x="engine-location", y="price", data=df)
plt.subplot(2, 2, 3)
sns.boxplot(x="drive-wheels", y="price", data=df, color='pink')
plt.subplot(2, 2, 4)
sns.boxplot(x="horsepower-binned", y="price", data=df, color='pink')
plt.suptitle("Categorical Variables")
# plt.show()

# We see that the distributions of price between the different body-style categories have a significant overlap, so body-style would not be a good predictor of price.
# The distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.
# Distribution of price between the different drive-wheels categories differs. As such, drive-wheels could potentially be a predictor of price.
# Hoursepower_binned could also be a good predictor for price.


# Descriptive Statistical Analysis
# The describe function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.
# The default setting of "describe" skips variables of type object.
print(df.describe(include=['object']))

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()  # "value_counts" only works on pandas series
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts)
# After examining the value counts of the engine location, we see that engine location would not be a good predictor variable for the price. This is because we only have three cars with a rear engine and 198 with an engine in the front, so this result is skewed.

print(df['drive-wheels'].unique())

print("Explore which type of drive wheel is most valuable.")
df_group_one = df[['drive-wheels', 'price']]
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).agg('mean')
print(df_group_one)

# From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel and front-wheel are approximately the same in price.

df_gptest = df[['drive-wheels', 'body-style', 'price']]
# grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
# # print(grouped_test1)
# grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot = df_gptest.groupby(['drive-wheels', 'body-style'], as_index=False).mean().pivot(index='drive-wheels',
                                                                                               columns='body-style')
grouped_pivot = grouped_pivot.fillna(0)  # fill missing values with 0
print(grouped_pivot)

# Use a heat map to visualize the relationship between Variables: Drive Wheels and Body Style vs. Price
# 'drive-wheel' and 'body-style' on the vertical and horizontal axis,
# plt.figure()
# plt.pcolor(grouped_pivot, cmap='RdBu')
# plt.colorbar()
# plt.show()

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
# move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
# rotate label if too long
plt.xticks(rotation=90)
fig.colorbar(im)
# plt.show()


# Explore the correlation of these variables with the car price.
# Pearson Correlation measures the linear dependence between two variables X and Y.
print("Calculate Pearson Correlation of all the 'int64' or 'float64' variables.")
print(df.corr(numeric_only=True))

# The P-value is the probability value that the correlation between these two variables is statistically significant.
# Significance level of 0.05 means that we are 95% confident that the correlation between the variables is significant.
"""
p-value is  <  0.001: we say there is strong evidence that the correlation is significant.
the p-value is  <  0.05: there is moderate evidence that the correlation is significant.
the p-value is  <  0.1: there is weak evidence that the correlation is significant.
the p-value is  >  0.1: there is no evidence that the correlation is significant."""

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print(f"The Pearson Correlation Coefficient is {pearson_coef:.2f},  with a P-value of P = {p_value}")
# Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant,
# although the linear relationship isn't extremely strong (~0.585).

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# Since the p-value is  <  0.001, the correlation between horsepower and price is statistically significant,
# and the linear relationship is quite strong (~0.809, close to 1).

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# Since the p-value is  <  0.001, the correlation between length and price is statistically significant,
# and the linear relationship is moderately strong (~0.691).

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
# Since the p-value is < 0.001, the correlation between width and price is statistically significant, and the linear
# relationship is quite strong (~0.751).

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# Since the p-value is  <  0.001, the correlation between curb-weight and price is statistically significant,
# and the linear relationship is quite strong (~0.834).

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
# Since the p-value is  <  0.001, the correlation between engine-size and price is statistically significant,
# and the linear relationship is very strong (~0.872).

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value)
# Since the p-value is  <  0.001, the correlation between bore and price is statistically significant, but the linear
# relationship is only moderate (~0.521).

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# Since the p-value is  <  0.001, the correlation between city-mpg and price is statistically significant,
# and the coefficient of about -0.687 shows that the relationship is negative and moderately strong.

pearson_coef, p_value = stats.pearsonr(df['highway-L/100km'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)
# Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant,
# and the coefficient of about -0.705 shows that the relationship is negative and moderately strong.


# ANOVA: Analysis of Variance The Analysis of Variance (ANOVA) is a statistical method used to test whether there are
# significant differences between the means of two or more groups. ANOVA returns two parameters: F-test score: ANOVA
# assumes the means of all groups are the same, calculates how much the actual means deviate from the assumption,
# and reports it as the F-test score. A larger score means there is a larger difference between the means. P-value:
# P-value tells how statistically significant our calculated score value is. If our price variable is strongly
# correlated with the variable we are analyzing, we expect ANOVA to return a sizeable F-test score and a small
# p-value.
# Chi-Square Test (Check if there is a relationship between variables) H0: variables are independent (if
# distributions are different)

# Explore if "drive wheels" impact "price"

grouped2 = df_gptest[['drive-wheels', 'price']].groupby('drive-wheels')
f_val, p_val = stats.f_oneway(grouped2.get_group('fwd')['price'], grouped2.get_group('rwd')['price'],
                              grouped2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped2.get_group('fwd')['price'], grouped2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

f_val, p_val = stats.f_oneway(grouped2.get_group('4wd')['price'], grouped2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# Three groups are correlated and highly correlated with each other, thus 'drive-wheels' impacts 'price'

# Important Variables:
# Continuous numerical variables:
# Length
# Width
# Curb-weight
# Engine-size
# Horsepower
# City-mpg
# highway-L/100km
# Wheel-base
# Bore

# Categorical variables:
# Drive-wheels


# Explore for multiple linear regression

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
plt.show()
plt.close()

# If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate.
# Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit.
# These residuals are not randomly spread around the x-axis, maybe a non-linear model is more appropriate.
