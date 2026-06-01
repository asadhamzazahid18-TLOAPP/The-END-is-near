import pandas as pd

pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')

print(pop)

print(pop.dtypes)

# the shape of the whole data frame

print(pop.shape)

# summary of statistic of Data frame using operations

print(pop.describe())

# info about the data frame like how many non null values are there and what is the data type of each column
print(pop.info())


# to get the column names of the data frame
print(pop.columns)

print(pop.head(10))

print(pop.tail(10))


# to get the company column of the data frame
company = pop['Company']
print('the column company will fetch the company names')
print(company)
print('-----------')

# to get the company and industry column of the data frame
company_industry = pop[['Company', 'Industry']]
print('company and its industry')
print(company_industry)
print('-----------')


# Using loc (label)
row = pop.loc[1]
print(row)

row2 = pop.loc[[2, 5]]
print(row2)

row3 = pop.loc[0:1]
print(row3)

#  to get the row where industry is technology
row4 = pop.loc[pop['Industry'] == 'Technology']
print(row4)

# get rows label from 0 to 1 and columns country and company type
row5 = pop.loc[:1, ['Country', 'Company_Type']]
print(row5)

# get rows label from 0 to 1 and columns from year to revenue usd
row6 = pop.loc[:1, 'Year':'Revenue_USD']
print(row6)

# get rows where industry is technology and columns uses ai and use case
row7 = pop.loc[pop['Industry'] == 'Technology', ['Uses_AI', 'Use_Case']]
print(row7)


# getting csv file again but this time index_col is year
# year   age name etc

df_index_col = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',', index_col='Year')


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

row11 = df_index_col.loc[[2025, 2020]]
print(row11)

# as year is our index and it's non unique value cause it's repeating
#  all over over so we have aplied boolean filtering so it doesn't crash

row12 = df_index_col.loc[(df_index_col.index >= 2020) & (
    df_index_col.index <= 2025), 'Uses_AI']
print(row12)

# to get all the rows and columns industry and country
row13 = df_index_col.loc[:, ['Industry', 'Country']]
print(row13)

# to get all the rows where country is USA and columns from company type to revenue usd
row14 = df_index_col.loc[df_index_col['Country']
                         == 'USA', 'Company_Type':'Revenue_USD']
print(row14)


# to get all the rows and columns from country to employee size
row15 = df_index_col.loc[:, 'Country':'Employee_Size']
print(row15)

# Using iloc (position)

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

# we find the lenth if length is 5 then index is 0 1 2 3 4
# so when adding new row we add at index 5 so it adds another one to the row
print(len(pop))

# to add a new row to the data frame
pop.loc[len(pop.index)] = [2030, 'manga', 'Products', 'zeeland', 'real',
                           '50', '-100000', 'No', 'Generative_AI', 93.00, 77]
print(pop)

print('------------------')

# to drop a row from the data frame
pop.drop(1, axis=0, inplace=False)

pop.drop([0, 3, 4], axis=0, inplace=False)

pop.drop(index=5, inplace=False)

print(pop)

# I wrote this line b4 cause I have to use columns later but I dropped earlier
# but then I inplace false so it doesn't change the original data frame
# pop = pd.read_csv(
#    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')

pop.drop('Uses_AI', axis=1, inplace=False)

pop.drop(columns='Company_Type', axis=1, inplace=False)

print(pop)

pop.rename(columns={'Year': 'Time'}, inplace=False)

# to rename multiple columns we can use mapper and axis 1 for columns and inplace true to change the original data frame
pop.rename(mapper={'Revenue_USD': 'Moneymoney',
           'Employee_Size': 'Loadsofpeople'}, axis=1, inplace=True)

print(pop)

pop.rename(index={1245: 9999}, inplace=True)

print(pop)
print(pop.columns.to_list())


# to filter the data frame where company is microsoft or ai roi percent is greater than 0.00
selected_rows = pop.query("Company == 'Microsoft' or AI_ROI_Percent > 0.00")

print(selected_rows.to_string())
print(len(selected_rows))

print(pop.columns)

sorted = pop.sort_values(by='Moneymoney')
print(sorted.to_string(index=False))

# to group the data frame by company and get the sum of loadsofpeople for each company
grouped = pop.groupby('Company')['Loadsofpeople'].sum()
print(grouped.to_string())
print(len(grouped))

print('_______________-')

pop_cleaned = pop.dropna()
pop.fillna(0, inplace=True)

print(pop)

# Creating an array
icecream = [1, 2, 3, 4, 5]

array9 = pd.array(icecream)
print(array9)

int_array = pd.array([1, 2, 4], dtype='int')
print(int_array)

print('_______________--')
