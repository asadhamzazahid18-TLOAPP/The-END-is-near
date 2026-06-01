from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

print(pop)

print(pop['Years_Experience'])
print(pop['Average_Salary'])
# Reshaping the data to be in the correct format for scikit-learn
x = pop['Years_Experience'].values.reshape(-1, 1)
y = pop['Average_Salary'].values.reshape(-1, 1)

print(x)
print(y)

print(x.shape)
print(x)
print(y.shape)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=45, train_size=0.7)

print(x_train)
print(y_train)


# Training the linear regression model
regressor = LinearRegression()

regressor.fit(x_train, y_train)

# Getting the intercept and coefficient of the linear regression model
print(regressor.intercept_)

print('_____________-----_____________')

print(regressor.coef_)


# Creating a function here to calculate the predicted salary based on the slope and intercept of the linear regression model and the years of experience
def calc(slope, intercept, Years_Experience):
    return slope*Years_Experience+intercept


# this and line 59 is same but different ways of doing it
result = calc(regressor.coef_[0], regressor.intercept_, 9.5)
print(result)

results = regressor.predict([[9.5]])
print(results)

# Predicting the salary for the test set
y_pred = regressor.predict(x_test)

# Creating a dataframe to compare the actual and predicted values of the test set
pop_preds = pd.DataFrame(
    {'Actual': y_test.squeeze(), 'Prediction': y_pred.squeeze()})
print(pop_preds)

# Calculating the mean absolute error, mean squared error, root mean squared error, and R2 score of the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation metrics
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')


# Now it's time for multiple regression
pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

le = LabelEncoder()

# Using Le Encoder this time to encode the categorical variables in the dataset
colall = ['Job_Title', 'Risk_Category', 'Education_Level']

for col in colall:
    pop[col + "_Encoded"] = le.fit_transform(pop[col])

x = pop[['Years_Experience', 'Risk_Category_                Encoded', 'Job_Title_Encoded', 'Education_Level_Encoded',
         'AI_Exposure_Index', 'Tech_Growth_Factor', 'Automation_Probability_2030']]
y = pop['Average_Salary']

print(x.shape)
print(y.shape)


# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=45, train_size=0.9)

print(x.shape)
print(x)

# Training and evaluating multiple regression models
models = {"Random Forest": RandomForestRegressor(),
          "Gradient Boost": GradientBoostingRegressor(),
          "Linear": LinearRegression(),
          "Huber": HuberRegressor()}


# Looping through the models, fitting them to the training data, making predictions on the test data, and printing the R² score and mean absolute error for each model

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{name}: R²={r2_score(y_test, y_pred):.3f} MAE={mean_absolute_error(y_test, y_pred):.0f}")


# # Compare actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n", results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# Calculating R² score manually
actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)
print(
    model.score(x_test, y_test))

print(f"\n{'!!!'*20}")
print('Another way of doing it')
print(f"\n{'(())(())(())'*20}")


models = {"Random Forest": RandomForestRegressor(),
          "Gradient Boost": GradientBoostingRegressor(),
          "Linear": LinearRegression(),
          "Huber": HuberRegressor()}


for name, model in models.items():

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mas = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    score = model.score(x_test, y_test)

    print(f"\n{'*-*'*12}")
    print(f"  Model : {name}")
    print(f"{'!!!!!'*12}")
    print(f"  R²    (r2_score) : {r2:.3f}")
    print(f"  R²    (score)    : {score:.3f}")
    print(f"  MAE              : {mas:.2f}")
    print(f"  MSE              : {mse:.2f}")
    print(f"  RMSE             : {rmse:.2f}")

# first one is exploratory/messy, second one is production ready.
# Each run values are differentiating but byfar gradient is giving the best value of all
