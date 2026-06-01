import pandas as pd

pop = pd.read_csv(
    'petal/ai_financial_market_daily_realistic_synthetic.csv', delimiter=',')

print(pop)

print(pop.dtypes)

# the shape of the whole data frame

print(pop.shape)

# summary of statistic of Data frame using operations

print(pop.describe())

# info about the data frame like non null values and memory usage
print(pop.info())

# printing the column names of the data frame
print(pop.columns)

# printing the first 10 rows of the data frame
print(pop.head(10))

# printing the last 10 rows of the data frame
print(pop.tail(10))


# priting Company column of the data frame
Company = pop['Company']
print('the company names are')
print(Company)
print('-----------')

# printing columns of company and dates
Company_date = pop[['Company', 'Date']]
print('company and dates are')
print(Company_date)
print('-----------')


# R&D_Spending_USD_Mn,AI_Revenue_USD_Mn,AI_Revenue_Growth_%,Event,Stock_Impact_%
# As these columns names are too long and I don't want to write it again and again inside the string
# so we are changing it right now

pop.rename(columns={'R&D_Spending_USD_Mn': 'Investment'}, inplace=True)
pop.rename(mapper={'AI_Revenue_USD_Mn': 'AI_Rev'}, axis=1, inplace=True)
pop.rename(columns={'AI_Revenue_Growth_%': 'Rev_growth'}, inplace=True)
pop.rename(mapper={'Stock_Impact_%': 'Stock_imp'}, axis=1, inplace=True)

print(pop.columns)

# selecting rows using loc (label)
row = pop.loc[1]
print(row)

row2 = pop.loc[[2, 5]]
print(row2)

row3 = pop.loc[0:1]
print(row3)

# fetching rows where company is openAI
row4 = pop.loc[pop['Company'] == 'OpenAI']
print(row4)

# selecting rows 0-1 and columns AI_Rev and Rev_growth
row5 = pop.loc[:1, ['AI_Rev', 'Rev_growth']]
print(row5)

# selecting rows 0-1 and columns from Date to Investment
row6 = pop.loc[:1, 'Date':'Investment']
print(row6)


# selecting rows where company is google and columns AI_Rev and Rev_growth
row7 = pop.loc[pop['Company'] == 'Google', [
    'AI_Rev', 'Rev_growth']]
print(row7)


# calling csv file again and setting index column as date
df_index_col = pd.read_csv(
    'petal/ai_financial_market_daily_realistic_synthetic.csv', delimiter=',', index_col='Date')

# renaming columns
df_index_col.rename(
    columns={'R&D_Spending_USD_Mn': 'Investment'}, inplace=True)
df_index_col.rename(
    mapper={'AI_Revenue_USD_Mn': 'AI_Rev'}, axis=1, inplace=True)
df_index_col.rename(
    columns={'AI_Revenue_Growth_%': 'Rev_growth'}, inplace=True)
df_index_col.rename(
    mapper={'Stock_Impact_%': 'Stock_imp'}, axis=1, inplace=True)


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

row11 = df_index_col.loc[['2015-01-02', '2023-08-12']]
print(row11)

# as date is our index and it's non unique value cause it's repeating
#  all over over so we have aplied boolean filtering

row12 = df_index_col.loc[(df_index_col.index >= '2023-08-15') & (
    df_index_col.index <= '2015-01-06'), 'Company']
print(row12)

# fetching rows where columns are AI_Rev and Rev_growth
row13 = df_index_col.loc[:, ['AI_Rev', 'Rev_growth']]
print(row13)

# calling rows where company is openAI and columns are from AI_Rev to Stock_imp
row14 = df_index_col.loc[df_index_col['Company']
                         == 'OpenAI', 'AI_Rev':'Stock_imp']
print(row14)


# fetching rows and columns from company to AI_Rev
row15 = df_index_col.loc[:, 'Company':'AI_Rev']
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

print(len(pop))


print('------------------')

# dropping rows and columns using drop function
pop.drop(1, axis=0, inplace=False)

pop.drop([0, 3, 4], axis=0, inplace=False)

pop.drop(index=5, inplace=False)

print(pop)


# dropping columns using drop function
pop.drop('AI_Rev', axis=1, inplace=False)

pop.drop(columns='Stock_imp', axis=1, inplace=False)

print(pop)

print(pop)

pop.rename(index={1245: 5999}, inplace=True)

print(pop)
print(pop.columns.to_list())

# filtering rows based on conditions using query function
selected_rows = pop.query(
    "Stock_imp >= 0.2 or Company == 'Meta' ")

print(selected_rows.to_string())
print(len(selected_rows))

print(pop.columns)

# sorting the data frame based on investment column
sorted = pop.sort_values(by='Investment')
print(sorted.to_string(index=False))

# groupping by row company and summing the AI_Rev column
grouped = pop.groupby('Company')['AI_Rev'].sum()
print(grouped.to_string())
print(len(grouped))

print('_______________-')

pop_cleaned = pop.dropna()
pop.fillna(0, inplace=True)

print(pop)

# creating an array
chips = [1, 2, 3, 4, 5]

array9 = pd.array(chips)
print(array9)

int_array = pd.array([1, 2, 4], dtype='int')
print(int_array)

print('_______________--')
