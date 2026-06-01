from collections import Counter
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')
print(pop)


# Enconding the categorical data to be able to use it in our models and algorithms
# Using OneHOTEncoder

cols = ['Job_Title', 'Education_Level']
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoding = encoder.fit_transform(pop[cols])


AI_df = pd.DataFrame(encoding, columns=encoder.get_feature_names_out(cols))

AI_df = AI_df.reset_index(drop=True)


X = pd.concat([pop[['Average_Salary', 'Years_Experience',
              'AI_Exposure_Index', 'Tech_Growth_Factor']].reset_index(drop=True), AI_df], axis=1)
x = X.copy()
y = le.fit_transform(pop['Risk_Category'])

print(x)
print(y)

# Scaling the data to make it easier for the models to learn and to make sure that all features are on the same scale and to improve the performance of the models
AIS = StandardScaler()
x_scaled = AIS.fit_transform(x)

# Using train test split to train the models and then see how better it learned
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, random_state=35, train_size=0.8)


# Different classification models
# and one or two ensemble models too

Lr = LogisticRegression(random_state=35)
svc = SVC(probability=True, random_state=35)
tree = DecisionTreeClassifier(random_state=35)
gr = GradientBoostingClassifier(random_state=35)
Gs = GaussianNB()
kg = KNeighborsClassifier()
ran = RandomForestClassifier(random_state=35)
vote = VotingClassifier(
    estimators=[
        ('lr', Lr),
        ('svc', svc),
        ('tree', tree),
        ('gb', gr),
        ('gnb', Gs),
        ('knn', kg),
        ('rf', ran)
    ],
    voting='soft'
)

# training here
Lr.fit(x_train, y_train)
svc.fit(x_train, y_train)
tree.fit(x_train, y_train)
gr.fit(x_train, y_train)
Gs.fit(x_train, y_train)
kg.fit(x_train, y_train)
ran.fit(x_train, y_train)
vote.fit(x_train, y_train)

print('-------------------')

# testing here models

Lr_pred = Lr.predict(x_test)
svc_pred = svc.predict(x_test)
tree_pred = tree.predict(x_test)
gr_pred = gr.predict(x_test)
Gs_pred = Gs.predict(x_test)
kg_pred = kg.predict(x_test)
ran_pred = ran.predict(x_test)
vote_pred = vote.predict(x_test)


model_pred = {
    'lr': Lr_pred,
    'svc': svc_pred,
    'tree': tree_pred,
    'gr': gr_pred,
    'Gs': Gs_pred,
    'kg': kg_pred,
    'ran': ran_pred,
    'vote': vote_pred
}

# Using loop to find the accuracy each model gave then see which one performed the best
for model, pred in model_pred.items():
    print(f"{model} Results:\n {classification_report(y_test, pred)}", sep="\n\n")


emoji_map = {
    "Low": "✅ Low Risk    — Safe",
    "Medium": "⚠️  Medium Risk — Monitor",
    "High": "❌ High Risk   — At Risk"
}


# this is the final verdict of the voting classifier which is the best one as it combines all the models and gives us the best result

votecounts = Counter(pred)
finalverdict = votecounts.most_common(1)[0][0]
finallabel = le.inverse_transform([finalverdict])[0]

print(f"  FINAL VERDICT: {emoji_map[finallabel]}")
print('----------------_______________--------------------')
# final verdict is that the job is at medium risk of being automated by 2030, so it's something to keep an eye on
# but not necessarily a cause for immediate concern. It's a good idea to stay informed about developments in AI and automation, and to consider upskilling or reskilling if you work in a field that's particularly vulnerable to automation.
