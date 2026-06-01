from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')

print(pop)

print(pop['AI_ROI_Percent'])
print(pop['AI_Maturity_Score'])
# Reshape the data for regression
x = pop['AI_Maturity_Score'].values.reshape(-1, 1)
y = pop['AI_ROI_Percent'].values.reshape(-1, 1)

print(x)
print(y)

print(x.shape)
print(x)
print(y.shape)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=45, train_size=0.7)

print(x_train)
print(y_train)


# Create and fit the linear regression model
regressor = LinearRegression()

regressor.fit(x_train, y_train)

print(regressor.intercept_)

print('_____________-----_____________')

print(regressor.coef_)


def calc(slope, intercept, score):
    return slope*score+intercept

# same line 54 and 56 are same but different ways
# one is with function and other is with predict method of regressor


results = calc(regressor.coef_, regressor.intercept_, 9.5)

results = regressor.predict([[9.5]])
print(results)

# Predict the target variable for the test set
y_pred = regressor.predict(x_test)

# Create a DataFrame to compare actual and predicted values
pop_preds = pd.DataFrame(
    {'Actual': y_test.squeeze(), 'Prediction': y_pred.squeeze()})
print(pop_preds)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
R2 = r2_score(y_test, y_pred)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {R2:.2f}')


# this is for multiple regression
pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')


cols = ['Industry', 'Company_Type', 'Employee_Size']

# One-Hot Encoding for categorical variables — removed Company and Country, too many unique values
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(pop[cols])

AI_pop = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
AI_pop = AI_pop.reset_index(drop=True)

# Added AI_Maturity_Score as it has the strongest correlation with AI_ROI_Percent
X = pd.concat(
    [pop[['Year', 'Revenue_USD', 'AI_Maturity_Score']].reset_index(drop=True), AI_pop], axis=1)

x = X
y = pop['AI_ROI_Percent']

print(x.shape)
print(y.shape)


# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=45, train_size=0.9)


print(x.shape)
print(x)

# Create and fit the linear regression model
regressor2 = LinearRegression()
pil = regressor2.fit(x_train, y_train)


# print the intercept and coefficients
print(regressor2.intercept_)
print(regressor2.coef_)

# Get the feature names from the DataFrame
features_names = x.columns

# Get the coefficients from the model
model_coeffeciciant = regressor2.coef_

# Create a DataFrame to display feature names and their corresponding coefficients
coefficients_pop = pd.DataFrame(data=model_coeffeciciant,
                                index=features_names, columns=['Coefficient_value'])


print(coefficients_pop)


y_pred = regressor2.predict(x_test)


# Create a DataFrame to compare actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n", results)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# Calculate R² score manually
actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)
print(
    regressor2.score(x_test, y_test))

print(
    f'R2 score for linear regression model was: {R2:.2f} and r2 score for multiple regression model was: {r2:.2f}')

# Conclusion comment
# Single and multiple regression both scored around 0.36-0.37
# confirming that AI_Maturity_Score is the only meaningful predictor
# and additional features add no value dataset has limited predictive signal
