import pandas as pd

# read the csv file and create a data frame

pop = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

print(pop)

# data types of each column

print(pop.dtypes)

# the shape of the whole data frame

print(pop.shape)

# summary of statistic of Data frame using operations

print(pop.describe())

# info about the data frame like non null values and memory usage
print(pop.info())

# fetching the column names of the data frame
print(pop.columns)

# fetching the first 10 rows of the data frame
print(pop.head(10))

# fetching the last 10 rows of the data frame
print(pop.tail(10))

# fetching the column job title of the data frame

Job_Tile = pop['Job_Title']
print('the column company will fetch the company names')
print(Job_Tile)
print('-----------')

# fetching the column job title and average salary of the data frame

JOB_title_avg_salary = pop[['Job_Title', 'Average_Salary']]
print('company and its industry')
print(JOB_title_avg_salary)
print('-----------')

# fetching the row and column of the data frame using loc
row = pop.loc[1]
print(row)

row2 = pop.loc[[2, 5]]
print(row2)

row3 = pop.loc[0:1]
print(row3)

# Fetch all rows where Job_Title is 'Teacher'
row4 = pop.loc[pop['Job_Title'] == 'Teacher']
print(row4)

# Fetch rows 0–1 for Years_Experience and Education_Level columns only
row5 = pop.loc[:1, ['Years_Experience', 'Education_Level']]
print(row5)

# fetch rows 0–1 for columns of Skill_1 to Skill_4
row6 = pop.loc[:1, 'Skill_1':'Skill_4']
print(row6)

# Fetch all rows where Education_Level is 'PHD' for columns Automation_Probability_2030 and Risk_Category
row7 = pop.loc[pop['Education_Level'] == 'PHD', [
    'Automation_Probability_2030', 'Risk_Category']]
print(row7)

# Loading same csv file again but this time index column is years of experience not like before random numbers
df_index_col = pd.read_csv(
    'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',', index_col='Years_Experience')

print(df_index_col)

# data types of each column
print(df_index_col.dtypes)

# the shape of the whole data frame
print(df_index_col.shape)


# summary of statistic of Data frame using operations
print(df_index_col.info())

# fetch rows 15 and 18 using loc from df_index_col
row11 = df_index_col.loc[[15, 18]]
print(row11)

# as year_experience is our index_col and it's non unique value cause it's repeating
# all over over so we have aplied boolean filtering so if it repeats it doesn't crash

# fetch rows from 2 to 30 for column risk category using loc from df_index_col

row12 = df_index_col.loc[(df_index_col.index >= 2) & (
    df_index_col.index <= 30), 'Risk_Category']
print(row12)


# fetch all rows for columns job title and education level
row13 = df_index_col.loc[:, ['Job_Title', 'Education_Level']]
print(row13)

# fetch all rows where education level is high school for columns job title to average salary
row14 = df_index_col.loc[df_index_col['Education_Level']
                         == 'High School', 'Job_Title':'Average_Salary']
print(row14)

# fetch all rows for columns job title to education level
row15 = df_index_col.loc[:, 'Job_Title':'Education_Level']
print(row15)

# Using iloc (positions based indexing) to fetch rows and columns from df_index_col
# NOT writing comment for each and everyone cause it's just positions instead of values but same splitting and concept
row16 = df_index_col.iloc[0]
print(row16)

row17 = df_index_col.iloc[0:10]
print(row17)

row18 = df_index_col.iloc[:1, :300]
print(row18)

row19 = df_index_col.iloc[[1, 3, 9]]
print(row19)

row20 = df_index_col.iloc[:, [3, 4]]
print(row20)

row21 = df_index_col.iloc[[1, 4, 6], 3:8]
print(row21)

row22 = df_index_col.iloc[[1, 2, 3,], [2, 3]]
print(row22)


print('-----------------------')


print(len(pop))

# adding a new row to the data frame using loc and len of the data frame
pop.loc[len(pop)] = ["LinkedIn", 100000, 1, 0.98, 2.35,
                     50.0, 'Low', 21.0, 35.6, 93.00, 77.0, 12.1, 13.2, 14.0, 15.00, 16.45, 17.20, 18.00]
print(pop)

print('------------------')

# dropping rows and columns from the data frame using drop method
pop.drop(1, axis=0, inplace=True)

pop.drop([0, 3, 4], axis=0, inplace=False)

pop.drop(index=5, inplace=False)

print(pop)

# I added this before cause I have to use columns again but then I inplace=false so values don't change in the main data so I commented this csv file part
# pop = pd.read_csv(
#  'project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')

pop.drop('Skill_10', axis=1, inplace=True)

pop.drop(columns='Job_Title', axis=1, inplace=True)

print(pop)

# renaming columns of the data frame using rename method
pop.rename(columns={'Average_Salary': 'moneymoney'}, inplace=True)

pop.rename(mapper={'Years_Experience': 'Timelearned',
           'Education_Level': 'Parentsfeesmeter'}, axis=1, inplace=True)

print(pop)

pop.rename(index={1245: 5999}, inplace=True)

print(pop)
print(pop.columns.to_list())

# filtering rows based on conditions using query method
selected_rows = pop.query(
    "Timelearned == 12 or AI_Exposure_Index > 0.20")

# to print the selected rows and also to print the length of the selected rows
print(selected_rows.to_string())
print(len(selected_rows))

print(pop.columns)

# sorting rows based on a column
sorted = pop.sort_values(by='moneymoney')
print(sorted.to_string(index=False))

# grouping rows based on a column and applying an aggregate function
grouped = pop.groupby('Parentsfeesmeter')['Tech_Growth_Factor'].sum()
print(grouped.to_string())
print(len(grouped))

print('_______________-')

pop_cleaned = pop.dropna()
pop.fillna(0, inplace=True)

print(pop)

# creating a pandas array from a list of chocolate and printing it
chocolate = [1, 2, 3, 4, 5]

array9 = pd.array(chocolate)
print(array9)

int_array = pd.array([1, 2, 4], dtype='int')
print(int_array)

print('_______________--')
print("Pandas over!!!")
