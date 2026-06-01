import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import pandas as pd

pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')


# -------PART 1 — Predicting AI_ROI_Percent-------
x = pop.select_dtypes(include='number').drop(
    columns=['AI_ROI_Percent'], errors='ignore')
y = pop['AI_ROI_Percent']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=22, train_size=0.9)

# Ensemble Learning Models
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
print("GradientBoosting - R2: ", r2_score(y_test, y_pred))

xgr = XGBRegressor()
xgr.fit(x_train, y_train)
y_pred2 = xgr.predict(x_test)
print("XGBoost - R2: ", r2_score(y_test, y_pred2))

adr = AdaBoostRegressor()
adr.fit(x_train, y_train)
y_pred3 = adr.predict(x_test)
print("AdaBoost - R2: ", r2_score(y_test, y_pred3))

cbr = CatBoostRegressor(iterations=100, depth=5,
                        learning_rate=0.01, loss_function='RMSE', verbose=0)
cbr.fit(x_train, y_train)
y_pred4 = cbr.predict(x_test)
print("CatBoost - R2: ", r2_score(y_test, y_pred4))

lgr = LGBMRegressor()
lgr.fit(x_train, y_train)
y_pred5 = lgr.predict(x_test)
print("LightGBM - R2: ", r2_score(y_test, y_pred5))

# Visualization of predictions vs actual values
fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(x=y_test, y=y_pred, label='GradientBoosting')
sns.lineplot(x=y_test, y=y_pred2, label='XGBoost')
sns.lineplot(x=y_test, y=y_pred3, label='AdaBoost')
sns.lineplot(x=y_test, y=y_pred4, label='CatBoost')
sns.lineplot(x=y_test, y=y_pred5, label='LightGBM')
ax.set_xlabel('y_test', color='g')
ax.set_ylabel('y_pred', color='g')
plt.title('Predicting AI_ROI_Percent')
plt.tight_layout()
plt.show()

# Dataset is synthetically generated with limited signal
# Models plateau around 25 because AI_ROI_Percent has no strong
# predictors beyond AI_Maturity_Score. This is a data limitation not a code issue


# -------PART 2 — Predicting Revenue_USD-------
x2 = pop.select_dtypes(include='number').drop(
    columns=['Revenue_USD'], errors='ignore')
y2 = pop['Revenue_USD']

# Train-Test Split
x_train2, x_test2, y_train2, y_test2 = train_test_split(
    x2, y2, random_state=22, train_size=0.9)

# Ensemble Learning Models
gbr2 = GradientBoostingRegressor()
gbr2.fit(x_train2, y_train2)
y_pred_r = gbr2.predict(x_test2)
print("\nGradientBoosting Revenue - R2: ", r2_score(y_test2, y_pred_r))

xgr2 = XGBRegressor()
xgr2.fit(x_train2, y_train2)
y_pred_r2 = xgr2.predict(x_test2)
print("XGBoost Revenue - R2: ", r2_score(y_test2, y_pred_r2))

adr2 = AdaBoostRegressor()
adr2.fit(x_train2, y_train2)
y_pred_r3 = adr2.predict(x_test2)
print("AdaBoost Revenue - R2: ", r2_score(y_test2, y_pred_r3))

cbr2 = CatBoostRegressor(iterations=100, depth=5,
                         learning_rate=0.01, loss_function='RMSE', verbose=0)
cbr2.fit(x_train2, y_train2)
y_pred_r4 = cbr2.predict(x_test2)
print("CatBoost Revenue - R2: ", r2_score(y_test2, y_pred_r4))

lgr2 = LGBMRegressor()
lgr2.fit(x_train2, y_train2)
y_pred_r5 = lgr2.predict(x_test2)
print("LightGBM Revenue - R2: ", r2_score(y_test2, y_pred_r5))

# Fix: added Bagging and ExtraTrees
bag = BaggingRegressor(n_estimators=100, random_state=42)
bag.fit(x_train2, y_train2)
y_pred_r6 = bag.predict(x_test2)
print("Bagging Revenue - R2: ", r2_score(y_test2, y_pred_r6))

etr = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr.fit(x_train2, y_train2)
y_pred_r7 = etr.predict(x_test2)
print("ExtraTrees Revenue - R2: ", r2_score(y_test2, y_pred_r7))

# Stacking model
stacking = StackingRegressor(
    estimators=[('gb', GradientBoostingRegressor()),
                ('xgb', XGBRegressor()),
                ('lgbm', LGBMRegressor())],
    final_estimator=Ridge()
)
stacking.fit(x_train2, y_train2)
y_pred_r8 = stacking.predict(x_test2)
print("Stacking Revenue - R2: ", r2_score(y_test2, y_pred_r8))

# Visualization of predictions vs actual values
fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(x=y_test2, y=y_pred_r, label='GradientBoosting')
sns.lineplot(x=y_test2, y=y_pred_r2, label='XGBoost')
sns.lineplot(x=y_test2, y=y_pred_r3, label='AdaBoost')
sns.lineplot(x=y_test2, y=y_pred_r4, label='CatBoost')
sns.lineplot(x=y_test2, y=y_pred_r5, label='LightGBM')
sns.lineplot(x=y_test2, y=y_pred_r6, label='Bagging')
sns.lineplot(x=y_test2, y=y_pred_r7, label='ExtraTrees')
sns.lineplot(x=y_test2, y=y_pred_r8, label='Stacking')
ax.set_xlabel('y_test', color='g')
ax.set_ylabel('y_pred', color='g')
plt.title('Predicting Revenue_USD')
plt.tight_layout()
plt.show()

print("---------------------")

# # Revenue prediction is significantly better than ROI , models successfully track major peaks
# XGBoost overshoots on high values while CatBoost and AdaBoost consistently underpredict
# GradientBoosting and Stacking are the most balanced performers overall
