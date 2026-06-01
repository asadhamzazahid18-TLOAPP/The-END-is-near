from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


pop = pd.read_csv('pot/top_10_ai_stocks.csv', delimiter=',')

sn.set(style='whitegrid')

# Fix: strip whitespace and convert to uppercase for consistent symbol formatting
pop['symbol'] = pop['symbol'].str.strip().str.upper()

print(pop["symbol"].nunique())
print(pop["symbol"].unique())

# Group by symbol and take the last entry for each symbol to get the most recent data
summary = pop.groupby("symbol").last().reset_index()

# Adding all 10 stocks name
watchlist = ['NVDA', 'META', 'ANET', 'AMZN',
             'PANW', 'NOW', 'AMD', 'PATH', 'TSLA', 'AI']

# Filter the original DataFrame to include only rows where the symbol is in the watchlist
filtered = pop[pop["symbol"].isin(watchlist)].copy()

print(filtered.shape)
print(filtered.head())

# Fix: equal sampling per symbol so no single stock dominates
# we are taking 5 random samples from each stock to create a more balanced dataset for the correlation heatmap, this way we can see the relationships between the metrics without one stock skewing the results
peep = filtered.groupby('symbol').apply(
    lambda x: x.sample(min(len(x), 5))
).reset_index(drop=True).sort_values(by='date')

g = sn.heatmap(peep.corr(numeric_only=True),
               cmap='coolwarm', annot=True, vmin=-1, vmax=1)
plt.show()

# from the heatmap we found out close and adjusted have the best relationships
# lET'S BE FAIR before when I used it only close and adjusted were related best but now other columns too Idk maybe coding magic hehe
# well lets go ahead

g = sn.scatterplot(data=filtered, x='open', y='close',
                   hue='symbol', alpha=0.7, sizes=(50, 1000), palette='bright')
plt.show()

# the above pattern is so cluttered and mixed so here is another try using facegrid

g = sn.FacetGrid(filtered, col="symbol", col_wrap=5, height=3, aspect=1)
g.map_dataframe(sn.scatterplot, x="open", y="close", alpha=0.6)
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.show()
# THIS IS better way of understanding as each graph is differently plotted instead of being thrown over each other like traffic

# Open and close prices move almost identically for every stock,
# meaning there's barely any intraday price movement. PATH stays the cheapest and tightest while NOW has the biggest price range overall.


# grouping by date and taking the mean of close price to find out which date has the highest close price
# on average across all stocks, this will help us to find out which date is the best for investing in AI stocks
top = pop.groupby('date')[
    "close"].mean().sort_values(ascending=False).head(70).reset_index()

#
sn.lineplot(peep, x='date', y='close', hue='symbol', errorbar='sd', marker='o')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# META and NOW have shown the most explosive growth over the years, with META climbing from nearly nothing to $500+. TSLA on the other hand is the wildest ride  spiking hard then
# crashing back down while everyone else grows steadily.

# finding range by subtracting low from high to find out the difference between the highest and lowest price of the day for each stock, this will help us to find out which stock is more volatile and which is less volatile
pop["range"] = pop["high"] - pop["low"]

sn.barplot(data=pop, x="range", y="symbol",
           orient="h", hue="symbol", palette="tab10")
plt.xlabel("Intraday range ($)")
plt.ylabel("Symbol")
plt.title("Intraday volatility by stock")
plt.show()

# NOW dominates with ~$8 intraday range,
# nearly 2x META and 40x NVDA, making it the clear volatility leader.
# The middle pack (TSLA, AI, PANW, ANET) clusters tightly around $2.5–$3.3, while NVDA's surprisingly subdued $0.2 range stands out as the anomaly.


sn.barplot(data=peep, x='symbol', y='close', hue='symbol', palette='tab10')
plt.title('Average Close Price by Stock')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print('-------------------')

# NOW leads avg close at ~$265 with the widest price dispersion,
# followed by PANW at ~$103 , both premium-priced enterprise names.
# NOW leads avg close at ~$265 with the widest price dispersion, followed by PANW at ~$103 , both premium-priced enterprise names.
# NVDA sits near zero, confirming its post-split adjusted price explains the massive volume but tiny intraday dollar range.


# instead of taking all data, we have applied this step to take out the high volume stocks
top = pop.groupby('symbol')[
    "volume"].mean().sort_values().reset_index()

# this barplot shows average trading volume per stock — higher volume means more active trading
sn.barplot(peep, x='symbol', y="volume", hue='symbol', palette="RdYlGn_r")
plt.show()

# NVDA towers at ~750M shares, 8x its nearest peers ,
# yet posts the tightest intraday range of the group.
# NOW pulls the opposite trick: maximum price swing, near-zero volume.

print('---------------')

cols = ["open", "high", "close", 'volume', "adjusted"]

# Fix: scale filtered not pop, and use .loc to avoid SettingWithCopyWarning
for col in cols:
    scaler = MinMaxScaler()
    filtered.loc[:, col] = scaler.fit_transform(filtered[[col]])

# Fix: melt filtered not pop, and use correct var_name
melted = filtered.melt(
    id_vars="symbol",
    value_vars=cols,
    var_name="Metric",   # fixed from "date"
    value_name="Score"
)

sn.lineplot(
    data=melted,
    x="symbol",
    y="Score",
    hue="Metric",
    errorbar=None
)
plt.title("How Key Metrics Shift Across Stocks")  # fixed title
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price metrics (open/high/close/adjusted) move in perfect lockstep,
# across all stocks, with volume flatlined near zero throughout.
# NOW and META spike as clear outliers, dominating normalized scores while the rest of the group clusters low.

print('-------------------')


# NVDA holds the volume, PANW/ANET sit expensive and ignored while NOW is basically the wildcard nobody asked for but everyone's watching.
# META's the only one pulling double duty; the rest (AMD, TSLA, AMZN, AI, PATH) are just... there.

print('-------------------')
