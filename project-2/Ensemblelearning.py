import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

colall = ['Job_Title', 'Education_Level']

# Encoding using le encoder
for col in colall:
    pop[col + "_Encoded"] = le.fit_transform(pop[col])


x = pop[['Job_Title_Encoded', 'Education_Level_Encoded', 'Years_Experience', 'Average_Salary',
         'AI_Exposure_Index', 'Tech_Growth_Factor']]

y = le.fit_transform(pop['Risk_Category'])

print(x)
print(y)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=22, train_size=0.9)


# Training and evaluating multiple ensemble learning models
gbr = GradientBoostingRegressor()

gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
print(r2_score(y_test, y_pred))


xgr = XGBRegressor()
xgr.fit(x_train, y_train)
y_pred2 = xgr.predict(x_test)
print("XGBoost - R2: ",
      r2_score(y_test, y_pred2))


adr = AdaBoostRegressor()
adr.fit(x_train, y_train)
y_pred3 = adr.predict(x_test)
print("AdaBoost - R2: ",
      r2_score(y_test, y_pred3))


cbr = CatBoostRegressor(iterations=100,
                        depth=5,
                        learning_rate=0.01,
                        loss_function='RMSE',
                        verbose=0)
cbr.fit(x_train, y_train)
y_pred4 = cbr.predict(x_test)
print("CatBoost - R2: ",
      r2_score(y_test, y_pred4))


lgr = LGBMRegressor()
lgr.fit(x_train, y_train)
y_pred5 = lgr.predict(x_test)
print("LightGBM - R2: ",
      r2_score(y_test, y_pred5))


# Visualizing the results

fig, ax = plt.subplots(figsize=(11, 5))

ax = sns.lineplot(x=y_test, y=y_pred,
                  label='GradientBoosting')
ax1 = sns.lineplot(x=y_test, y=y_pred2,
                   label='XGBoost')
ax2 = sns.lineplot(x=y_test, y=y_pred3,
                   label='AdaBoost')
ax3 = sns.lineplot(x=y_test, y=y_pred4,
                   label='CatBoost')
ax4 = sns.lineplot(x=y_test, y=y_pred5,
                   label='LightGBM')

ax.set_xlabel('y_test', color='g')
ax.set_ylabel('y_pred', color='g')
plt.show()

wait = input("wait for....")

# GradientBoosting, XGBoost and LightGBM are almost perfectly predicting actual values, making them the clear winners here.
# AdaBoost and CatBoost both struggle, especially at the higher end where they start falling behind the actual values.
