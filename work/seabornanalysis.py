import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')


le = LabelEncoder()
pop['Employee_Size'] = le.fit_transform(pop['Employee_Size'])

# instead of taking all data, we have applied this step to take out the high quality industries only revenue,


top = pop.groupby(['Industry'])["Revenue_USD"].sum().sort_values(
    ascending=False).head(100).reset_index()
# filtering it now so only the best industries are here

filtered_df = pop[pop["Industry"].isin(top['Industry'])]


sn.set(style='whitegrid')

sn.barplot(data=top, x='Industry', y="Revenue_USD")
plt.show()
# Technology leads revenue by a massive margin, nearly doubling every other industry, while Consumer Goods, Automotive, and Industrial sit far behind at the bottom.

# It keeps 8 max rows per industry to avoid overcrowding on one industry
peep = filtered_df.groupby('Industry').apply(
    lambda x: x.sample(min(len(x), 8))
).reset_index(drop=True).copy()


# as employee size is categorical data, we have to encode it to be able to use it in the scatter plot, and we have done t
# print(pop['Employee_Size'].value_counts())
# g = sn.scatterplot(data=peep, x='Employee_Size', y='Revenue_USD',
#                   hue='Industry', alpha=0.7, size='AI_Maturity_Score', sizes=(50, 1000))
# plt.show()

g = sn.catplot(data=peep, x='Year', y='Revenue_USD',
               kind='bar', hue='Industry')
plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Technology is the only industry that shows consistent growth from 2020 to 2025,
# while Retail had the most dramatic rise and fall. Overall 2024 was a down year for almost every industry across the board.

# Remove sn.set(style=eval) completely from your code first

sn.barplot(data=pop, x="Industry", y="Revenue_USD",
           hue="Year", errorbar=None)
plt.xticks(rotation=45)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
print('-------------------')
# we can see only three industries dominate while others sit flat at the bottom
# there is some data limitation cause we took limited data so
# there are some cautions to keep in mind while reading this chart

print('---------------')

g = sn.barplot(data=peep, x="Industry", y="Revenue_USD", hue='Industry', )

plt.show()

# E-commerce and Technology lead in average revenue,
# while the rest of the industries sit closely together with no major differences. The large error bars across all industries suggest revenue varies hugely within each sector, so the averages can be misleading.

print('---------------')


# here we are finding the values relations/intensity
numeric_cols = pop.select_dtypes(include='number')
sn.heatmap(peep.corr(numeric_only=True), cmap='coolwarm', annot=True)
plt.show()

# we find out here the strongest relation is between AI_maturity_score and AI_ROI_Percent
# which was about 59.00


sn.lineplot(peep, x='AI_Maturity_Score', y='AI_ROI_Percent',
            label='comparing the values', color='red', markers='o')

plt.show()

# AI ROI stays at zero until maturity hits 40,
# then rises sharply, showing that only companies with serious AI investment start seeing real returns.
# Beyond 70 the returns get volatile, meaning high maturity doesn't always guarantee consistent gains.


print('-------------------')
print("Overall from all graphs above we found out")
# Overall the data tells us that job title, AI maturity, and industry type are the three
# biggest factors driving salary, automation risk, and revenue respectively.
# Higher risk doesn't always mean higher pay, more AI maturity does lead to better ROI,
# and Technology dominates every other industry in revenue — but data limitations
# across both datasets mean these findings should be taken as directional not definitive.
