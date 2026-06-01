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

pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')
print(pop)


# One-Hot Encoding for categorical variables
cols = ['Company', 'Country', 'Company_Type', 'Employee_Size']
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoding = encoder.fit_transform(pop[cols])

AI_df = pd.DataFrame(encoding, columns=encoder.get_feature_names_out(cols))

AI_df = AI_df.reset_index(drop=True)


X = pd.concat(
    [pop[['Year', 'Revenue_USD']].reset_index(drop=True), AI_df], axis=1)
x = X.copy()
y = pop['Uses_AI']

print(x)
print(y)

# Feature Scaling
AIS = StandardScaler()
x_scaled = AIS.fit_transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, random_state=35, train_size=0.8)


# Model Building
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

# Model Training
Lr.fit(x_train, y_train)
svc.fit(x_train, y_train)
tree.fit(x_train, y_train)
gr.fit(x_train, y_train)
Gs.fit(x_train, y_train)
kg.fit(x_train, y_train)
ran.fit(x_train, y_train)
vote.fit(x_train, y_train)

print('-------------------')

# Model Evaluation
Lr_pred = Lr.predict(x_test)
svc_pred = svc.predict(x_test)
tree_pred = tree.predict(x_test)
gr_pred = gr.predict(x_test)
Gs_pred = Gs.predict(x_test)
kg_pred = kg.predict(x_test)
ran_pred = ran.predict(x_test)
vote_pred = vote.predict(x_test)


# Print classification reports for each model
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

# Display classification reports for each model
for model, pred in model_pred.items():
    print(f"{model} Results:\n {classification_report(y_test, pred)}", sep="\n\n")


# Final Verdict based on Voting Classifier
vote_pred_numeric = [1 if i == "Yes" else 0 for i in vote_pred]

final_verdict = "YES✅ " if np.mean(vote_pred_numeric) > 0.5 else "NO ❌"
print("FINAL VERDICT:", final_verdict)
