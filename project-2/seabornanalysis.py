from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

# Using seaborn to see how data is connected to each other and how they are related to each other
# Setting design to whitegrid for better visibility of the plots and graphs
sn.set(style='whitegrid')

# We will be using heatmap, scatterplot, lineplot, violinplot and barplot to visualize the data

# all rows and clumns from job title to risk category cause they are the only values we need and rest are not the high quality for our data experiments here
peep = pop.loc[:, 'Job_Title':'Risk_Category'].copy()

# First using heatmap cause it tells directly out of all data which values affect each other the strongest
g = sn.heatmap(peep.corr(numeric_only=True),
               cmap='coolwarm', annot=True, vmin=-1, vmax=1)
plt.show()
# Heatmap shows that Years experience and AI exposure index are are affecting each other the most with a value of 0.035
# SO this is our strongest relationship here out of all but it's consider no relationship as it's  very close to 0 so this is a weak dataset

# Using Scatterplot to see how education level affects the salary and how experience affects the salary and how they are related to each other

g = sn.scatterplot(data=peep, x='Years_Experience', y='Average_Salary',
                   hue='Education_Level', alpha=0.7, sizes=(50, 1000))
plt.show()
# Scatterplot showed that education level has no effect on the salary

# Grouping to resolve non-unique combinations of Experience & Education Level
# and calculate mean salary for each unique combination
# we took only 50 rows cause we want the best quality data and ascending false means we get better quality data(10,9,8)desecnding order (not 1,2,3)
# reset index means if in between process of grouping and sorting the index gets messed up then it will reset it to default 0,1,2,3,4,5,6,7,8,9

top = pop.groupby(['Years_Experience', 'Education_Level'])[
    "Average_Salary"].mean().sort_values(ascending=False).head(50).reset_index()


# Using lineplot
sn.lineplot(top, x='Years_Experience',
            y='Average_Salary', hue='Education_Level', errorbar='sd')
# Still the relationship is weak as given b4 value of 0.017 which means no relationship

plt.show()

# using violinplot
g = sn.FacetGrid(peep, col='Job_Title', col_wrap=5, height=3)
g.map(sn.violinplot, 'Risk_Category', 'Average_Salary',
      order=['Low', 'Medium', 'High'])
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()

# Higher risk doesn't always mean higher pay. technical roles like AI engineers and doctors earn more as risk increases, but service roles like nurses and teachers barely see a difference. At the end of the day, Your field matters more than your risk level


# Using lineplot again but with different columns
g = sn.lineplot(data=peep, x="Education_Level", y="Average_Salary",
                marker="o", hue='Job_Title')

plt.show()

print('-------------------')


# instead of taking all data, we have applied this step to take out the high quality industries only revenue,


top = pop.groupby(['Job_Title'])[
    "Automation_Probability_2030"].mean().sort_values(ascending=False).reset_index()
# filtering it now so only the best industries are here


# Using barplot
sn.barplot(top, x='Automation_Probability_2030', y="Job_Title",
           palette="RdYlGn_r")
plt.show()

# this barplot clearly shows that how many jobs are at the danger of being replaced by AI robots or other systems
# This graph shows clearly how AI will impact the industries clearly

print('---------------')


# Here a technique is used called Melt (not your heart) but a technique which helps easier to digest more columns
# you can't do y=["AI_Exposure_Index"or"Tech_Growth_Factor"or"Automation_Probability_2030"] but
# we can use melt and make them into a neat columns and before this one thing is that these must be standard scarlarized cause
# else three columns will be on three different scales and will cause trouble


cols = ["AI_Exposure_Index",
        "Tech_Growth_Factor",
        "Automation_Probability_2030"]

pop[cols] = MinMaxScaler().fit_transform(pop[cols])

melted = pop.melt(
    id_vars="Years_Experience",
    value_vars=cols,
    var_name="Metric",
    value_name="Score"
)

sn.lineplot(
    data=melted,
    x="Years_Experience",
    y="Score",
    hue="Metric",
    errorbar=None
)
plt.xticks(rotation=45)
plt.title("How Key Metrics Shift with Experience")
plt.show()
print('-------------------')

# Experience level don't really seem to have much impact on
# AI exposure, tech growth, or automation probability;
# all three metrics just stay flat the whole way through. Honestly it's probably just a weak dataset, not something you'd actually see in the real world.
