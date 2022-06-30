# Explore the Data

import pandas as pd
import numpy as np

# load the training dataset
data_bikes = 'data.csv'
bike_data = pd.read_csv(data_bikes)
print(bike_data.head())

# For example, let's add a new column named day to the dataframe by extracting the day component from the existing
# dteday column.

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
bike_data.head(32)
print(bike_data.head())

# We can use the dataframe's describe method to generate these for the numeric features as well as the rentals
# label column.
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
print(bike_data[numeric_features + ['rentals']].describe())

import matplotlib.pyplot as plt

# This ensures plots are displayed inline in the Jupyter notebook
# %matplotlib inline

# Get the label column
label = bike_data['rentals']


# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (12,12))

# Plot the histogram
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')

# Show the figure
fig.show()

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
plt.show()

# plot a bar plot for each categorical feature count
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
plt.show()


# Now that we know something about the distribution of the data in our columns, we can start to look for relationships
# between the features and the rentals label we want to be able to predict.

for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
plt.show()

# Train a Regression Model

# Now that we've explored the data, it's time to use it to train a regression
# model that uses the features we've identified as potentially predictive
# to predict the rentals label. The first thing we need to do is to separate the
# features we want to use to train the model from the label we want it to predict.

# Separate features and labels
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

# To randomly split the data, we'll use the train_test_split function in the scikit-learn library. This library is one
# of the most widely used machine learning packages for Python.


from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# In Scikit-Learn, training algorithms are encapsulated in estimators, and in this case we'll use the LinearRegression
# estimator to train a linear regression model.
from sklearn.linear_model import LinearRegression

# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)
print (model)
# Evaluate the Trained Model

predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])

# Let's see if we can get a better indication by visualizing a scatter plot that compares the predictions to
# the actual labels.

plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

"""
You can quantify the residuals by calculating a number of commonly used evaluation metrics. We'll focus on the following three:

Mean Square Error (MSE): The mean of the squared differences between predicted and actual values. This yields a relative
 metric in which the smaller the value, the better the fit of the model
Root Mean Square Error (RMSE): The square root of the MSE. This yields an absolute metric in the same unit as the label 
(in this case, numbers of rentals). The smaller the value, the better the model (in a simplistic sense, it represents 
the average number of rentals by which the predictions are wrong!)
Coefficient of Determination (usually known as R-squared or R2): A relative metric in which the higher the value, the 
better the fit of the model. In essence, this metric represents how much of the variance between predicted and actual 
label values the model is able to explain.
"""

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)
print("R2:", r2)

