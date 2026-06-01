import pandas as pd

# importing the data from csv file and creating a data frame
pop = pd.read_csv(
    'pot/top_10_ai_stocks.csv', delimiter=',')

print(pop)

print(pop.dtypes)

# the shape of the whole data frame

print(pop.shape)

# summary of statistic of Data frame using operations

print(pop.describe())

# info about the data frame like number of non null values and data types of each column
print(pop.info())

# accessing the columns of the data frame
print(pop.columns)

# accessing the first 10 rows of the data frame
print(pop.head(10))

# accessing the last 10 rows of the data frame
print(pop.tail(10))


# accessing the symbol column of the data frame
symbol = pop['symbol']
print('the company names are')
print(symbol)
print('-----------')

# accessing the symbol and date column of the data frame
symbol_date = pop[['symbol', 'date']]
print('company and dates are')
print(symbol_date)
print('-----------')

# accessing the symbol and date column of the data frame using loc
row = pop.loc[1]
print(row)

row2 = pop.loc[[2, 5]]
print(row2)

row3 = pop.loc[0:1]
print(row3)

# accessing the row where symbol is NVDA using loc
row4 = pop.loc[pop['symbol'] == 'NVDA']
print(row4)

# fetch row 0-1 and columns open and high using loc
row5 = pop.loc[:1, ['open', 'high']]
print(row5)

# fetch row 0-1 and columns open to close using loc
row6 = pop.loc[:1, 'open':'close']
print(row6)

# fetch row where symbol is AMZN and columns volume and adjusted using loc
row7 = pop.loc[pop['symbol'] == 'AMZN', [
    'volume', 'adjusted']]
print(row7)


# again calling csv file this time with index column as volume
df_index_col = pd.read_csv(
    'pot/top_10_ai_stocks.csv', delimiter=',', index_col='volume')


print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

row11 = df_index_col.loc[[2714688000, 154704000]]
print(row11)

# as year_experience is our index and it's non unique value cause it's repeating
#  all over over so we have aplied boolean filtering to avoid crashing

row12 = df_index_col.loc[(df_index_col.index >= 2) & (
    df_index_col.index <= 30), 'adjusted']
print(row12)

# fetching all the rows and columns high and low using loc
row13 = df_index_col.loc[:, ['high', 'low']]
print(row13)

# fetching the row where symbol is META and columns from high to adjusted using loc
row14 = df_index_col.loc[df_index_col['symbol']
                         == 'META', 'high':'adjusted']
print(row14)


# fetching all the rows and columns from symbol to open using loc
row15 = df_index_col.loc[:, 'symbol':'open']
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

pop.loc[len(pop.index)] = ["ASHU", "1998-02-27",
                           21.5, 1110, 567, 423425, 32432, 87657]
print(pop)

print('------------------')

pop.drop(1, axis=0, inplace=True)

pop.drop([0, 3, 4], axis=0, inplace=True)

pop.drop(index=5, inplace=True)

print(pop)

# calling csv file again as B4 we drop columns and rows and inplace true
# so we called it again
pop = pd.read_csv(
    'pot/top_10_ai_stocks.csv', delimiter=',')

pop.drop('adjusted', axis=1, inplace=False)

pop.drop(columns='low', axis=1, inplace=True)

print(pop)

# renaming the column name symbol to TheOGS using rename function and inplace true
pop.rename(columns={'symbol': 'TheOGS'}, inplace=True)

# renaming the column name adjusted to fixed and date to calendar using rename function and inplace true
pop.rename(mapper={'adjusted': 'fixed',
           'date': 'calendar'}, axis=1, inplace=True)

print(pop)

pop.rename(index={1245: 5999}, inplace=True)

print(pop)
print(pop.columns.to_list())


# filtering the data frame using query function to select the rows where TheOGS is META or close is greater than 0.050000
selected_rows = pop.query(
    "TheOGS =='META' or close > 0.050000")

print(selected_rows.to_string())
print(len(selected_rows))

print(pop.columns)

# sorting values
sorted2 = pop.sort_values(by='calendar', ascending=True)
print(sorted2.to_string(index=False))

# groupping values by rows TheOGS and calculating the sum of fixed column for each group

grouped = pop.groupby('TheOGS')['fixed'].sum()
print(grouped.to_string())
print(len(grouped))

print('_______________-')

pop_cleaned = pop.dropna()
pop.fillna(0, inplace=True)

print(pop)

# creating a pandas array from a list of chips
chips = [1, 2, 3, 4, 5]

array9 = pd.array(chips)
print(array9)

int_array = pd.array([1, 2, 4], dtype='int')
print(int_array)

print('_______________--')
