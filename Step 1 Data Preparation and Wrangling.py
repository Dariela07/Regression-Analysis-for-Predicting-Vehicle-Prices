import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pylab as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
# df = pd.read_csv(url, header=None)
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]

# df = pd.read_csv(filename, names = headers)

df.columns = headers
df = df.replace('?', np.NaN)  # df.replace("?", np.nan, inplace = True)

# This method will provide various summary statistics, excluding NaN (Not a Number) values.
# df.describe()  # This shows the statistical summary of all numeric-typed (int, float) columns.
# print(df.describe(include="all")) # check all the columns including those that are of type object
print(df.head())

# 2 DATA WRANGLING: converting data from the initial format to a format that may be better for analysis.
# Deal with missing data
missing_data = df.isnull()
missing_data2 = df.notnull()

# for column in missing_data.columns.values.tolist():  # .columns --> index, .values --> array (object), .tolist() --> list
#     print(missing_data[column].value_counts())
#     print("")

# Whole columns should be dropped only if most entries in the column are empty.

# Drop the whole row:
# To predict price, any row now without price data is not useful
df = df.dropna(subset=["price"], axis=0)  # drop missing values along the column "price
# reset index, because we dropped two rows
df.reset_index(drop=True, inplace=True)

# Replace by mean:
# For clarity, specify axis='index' (instead of axis=0) or axis='columns' (instead of axis=1).
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
avg_bore = df['bore'].astype('float').mean(axis=0)
stroke_everage = df['stroke'].astype('float').mean(axis='index')
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
df["normalized-losses"] = df["normalized-losses"].replace(np.nan, avg_norm_loss)
df["bore"] = df["bore"].replace(np.nan, avg_bore)
df['stroke'] = df['stroke'].replace(np.nan, stroke_everage)
df['horsepower'] = df['horsepower'].replace(np.nan, avg_horsepower)
df['peak-rpm'] = df['peak-rpm'].replace(np.nan, avg_peakrpm)

# Replace by frequency:
# Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur
df['num-of-doors'].value_counts()  # all values
# ".idxmax()": calculate the most common type
df["num-of-doors"] = df["num-of-doors"].replace(np.nan, df['num-of-doors'].value_counts().idxmax())

# Correct data format
#  Numerical variables: type 'float' or 'int', and variables with strings such as categories: type 'object'.
df[["bore", "stroke", "peak-rpm", "price"]] = df[["bore", "stroke", "peak-rpm", "price"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
# print(df.dtypes)

# Data Standardization:
# transforming data into a common format, to make the meaningful comparison.
# Also a term for a type of data normalization where we subtract the mean and divide by the standard deviation.
df['city-L/100km'] = 235 / df["city-mpg"]  # Convert mpg to L/100km by mathematical operation
df.rename(columns={'highway-mpg': 'highway-L/100km'}, inplace=True)
df['highway-L/100km'] = 235 / df['highway-L/100km']

# Data Normalization
# normalize those variables so their value ranges from 0 to 1
df['length'] = df['length'] / df['length'].max()
df['width'] = df['width'] / df['width'].max()
df['height'] = df['height'] / (df['height'].max())

# Binning
# Transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis
df["horsepower"] = df["horsepower"].astype(int, copy=True)
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)
# print(df[['horsepower','horsepower-binned']].head(20))
# print(df["horsepower-binned"].value_counts())
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df["horsepower"].plot(kind='hist', xlabel='horsepower', ylabel='count', title='horsepower bin')
plt.subplot(1, 2, 2)
df["horsepower-binned"].value_counts().plot(kind='bar', xlabel='horsepower', ylabel='count', title='horsepower bin')
# plt.show()

# Indicator Variable (or Dummy Variable)
# To label categories: don't have inherent meaning
# We can use categorical variables for regression analysis, as regression doesn't understand words, only numbers.
dummy_variable_1 = pd.get_dummies(df["fuel-type"], dtype='int64')  # 2 columns each with a name form row
# dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis='columns')
df.drop("fuel-type", axis=1, inplace=True)

# #  indicator variable for the column "aspiration"
# new_v = pd.get_dummies(df["aspiration"])
# df = pd.concat([df, new_v], axis=1)
# df.rename(columns={'std': 'aspiration_std', 'turbo': 'aspiration_turbo'}, inplace=True)
# df.drop(['aspiration'], axis=1, inplace=True)

print(df.shape)
print(df.head())
df.to_csv("clean_data.csv", index=False)


