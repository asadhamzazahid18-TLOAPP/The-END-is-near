from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# going little different direction this seaborn

# Global style
plt.rcParams.update({
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': False,
    'axes.grid':          True,
    'grid.color':         '#e8e8e8',
    'grid.linewidth':     0.6,
    'axes.facecolor':     '#fafafa',
    'figure.facecolor':   '#ffffff',
    'xtick.color':        '#888888',
    'ytick.color':        '#888888',
})

# Reusable trend line function


def add_trendline(ax, x, y, color='#333333'):
    mask = x.notna() & y.notna()
    x_clean = x[mask]
    y_clean = y[mask]
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
    ax.plot(x_line, p(x_line),
            color=color, linewidth=1.5,
            linestyle='--', alpha=0.5, label='Trend')


# Load data
pop = pd.read_csv(
    'petal/ai_financial_market_daily_realistic_synthetic.csv', delimiter=',')

# groupping company rows and sum of AI_Revenue_Growth_%
print(pop.groupby('Company')['AI_Revenue_Growth_%'].nunique())

# Check if rows are being duplicated
print(pop.groupby('Company').size())

# See the actual unique values
print(pop[pop['Company'] == 'Google']['AI_Revenue_Growth_%'].unique())


# Convert 'Date' to datetime and sort by date
pop['Date'] = pd.to_datetime(pop['Date'])
pop = pop.sort_values('Date')

print(pop["Company"].nunique())
print(pop["Company"].unique())

# ── Filter to 3 companies ────────────────────────────────────────────────────
filtered = pop[pop["Company"].isin(['OpenAI', 'Google', 'Meta'])].copy()

print(filtered.shape)
print(filtered.head())

# ============================================================
# GRAPH 1 — Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(filtered.corr(numeric_only=True),
            cmap='coolwarm', annot=True,
            vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            fmt='.2f')
ax.set_title('Feature Correlation Heatmap',
             fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# R&D Spending and AI Revenue move in near-perfect lockstep (0.94),
# confirming R&D directly fuels revenue. Stock Impact is completely decorrelated
# from everything (~0.00)  market price moves to its own beat regardless of AI fundamentals.

# ============================================================
# GRAPH 2 — Scatter: R&D Spending vs AI Revenue with trend line
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered, x='R&D_Spending_USD_Mn',
                y='AI_Revenue_USD_Mn',
                hue='Company', alpha=0.7, palette='bright', ax=ax)

# Add trend line using our reusable function
add_trendline(ax, filtered['R&D_Spending_USD_Mn'],
              filtered['AI_Revenue_USD_Mn'], color='#333333')

# Add company name labels on cluster centres
for company in ['OpenAI', 'Google', 'Meta']:
    subset = filtered[filtered['Company'] == company]
    ax.annotate(company,
                xy=(subset['R&D_Spending_USD_Mn'].mean(),
                    subset['AI_Revenue_USD_Mn'].mean()),
                fontsize=10, fontweight='bold', ha='center',
                color='#1a1a1a')

ax.set_title('R&D Spending vs AI Revenue by Company',
             fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xlabel('R&D Spending (USD Mn)', fontsize=10,
              color='#555555', labelpad=8)
ax.set_ylabel('AI Revenue (USD Mn)', fontsize=10,
              color='#555555', labelpad=8)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('scatter_rd_revenue.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
# The scatter shows each company occupies a distinct cluster,
# more R&D spending clearly leads to higher AI revenue

# Google dominates spend and revenue staying on trend, Meta beats it per dollar, OpenAI is barely off the ground. Three very different stages of AI monetization…


# ============================================================
# GRAPH 3 — FacetGrid: AI Revenue Growth vs Stock Impact
# ============================================================
# Remove outliers so main cluster is visible
q99 = filtered['Stock_Impact_%'].quantile(0.99)
filtered_clean = filtered[filtered['Stock_Impact_%'] <= q99].copy()

g = sns.FacetGrid(filtered_clean, col="Company",
                  col_wrap=3, height=4, aspect=1.1)
g.map_dataframe(sns.regplot,
                x="AI_Revenue_Growth_%",
                y="Stock_Impact_%",
                scatter_kws={'alpha': 0.4, 's': 15, 'color': '#2c7bb6'},
                line_kws={'color': '#e74c3c', 'linewidth': 1.5})
g.set_titles(col_template="{col_name}", size=11, fontweight='bold')
g.set_axis_labels("AI Revenue Growth (%)", "Stock Impact (%)")
plt.tight_layout()
plt.savefig('facetgrid_growth_impact.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()

# The flat trend suggests no correlation,
# but the vertical stripe pattern reveals the data is likely binned or synthetic ; so this chart reflects a data limitation more than a real market truth.


# ============================================================
# GRAPH 4 — Line chart: Stock Impact over time
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))

colors_map = {'OpenAI': '#2c7bb6', 'Google': '#f46d43', 'Meta': '#1a9641'}

for company, color in colors_map.items():
    subset = filtered[filtered['Company'] == company].sort_values('Date')
    ax.plot(subset['Date'], subset['Stock_Impact_%'],
            color=color, linewidth=0.8, alpha=0.75, label=company)

# Zero reference line
ax.axhline(y=0, color='#333333', linewidth=0.8,
           linestyle='--', alpha=0.4)

# Fix date axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right', fontsize=9)

ax.set_title('Stock Impact Over Time — OpenAI vs Google vs Meta',
             fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xlabel('Date', fontsize=10, color='#555555', labelpad=8)
ax.set_ylabel('Stock Impact (%)', fontsize=10, color='#555555', labelpad=8)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('stock_impact_time.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()

# OpenAI's spikes explode post-2022 (up to ~17%) while Google and Meta stay flat throughout, the ChatGPT era clearly split OpenAI's volatility from the pack.

# ============================================================
# GRAPH 5 — Bar chart: Average Stock Impact per Company
# ============================================================
impact_summary = filtered.groupby('Company')[
    'Stock_Impact_%'].mean().sort_values().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(impact_summary['Company'],
               impact_summary['Stock_Impact_%'],
               color=['#2c7bb6', '#f46d43', '#1a9641'],
               edgecolor='white', height=0.5)

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{width:.3f}%', va='center', fontsize=10, color='#333333')

ax.set_title('Average Stock Impact by Company',
             fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xlabel('Mean Stock Impact (%)', fontsize=10,
              color='#555555', labelpad=8)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('barplot_impact.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()

# OpenAI edges ahead at ~0.040% vs Google's 0.026% and Meta's 0.010%, but all three are so close to zero the differences are meaningless in practice.


# ============================================================
# GRAPH 6 — Normalised line chart: How metrics compare
# ============================================================
cols = ["R&D_Spending_USD_Mn",
        "AI_Revenue_USD_Mn",
        "AI_Revenue_Growth_%",
        "Stock_Impact_%"]

# Scale each column independently between 0 and 1
filtered_scaled = filtered.copy()
for col in cols:
    scaler = MinMaxScaler()
    filtered_scaled[col] = scaler.fit_transform(filtered[[col]])

# Melt using filtered_scaled, not pop
melted = filtered_scaled.melt(
    id_vars="Company",
    value_vars=cols,
    var_name="Metric",
    value_name="Normalised_Value"
)

fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(
    data=melted,
    x="Metric",
    y="Normalised_Value",
    hue="Company",
    errorbar=None,
    palette='bright',
    linewidth=2,
    ax=ax
)
ax.set_title('Normalised Metric Comparison by Company',
             fontsize=13, fontweight='bold', loc='left', pad=12)
ax.set_xlabel('', fontsize=10)
ax.set_ylabel('Normalised Value (0–1)', fontsize=10,
              color='#555555', labelpad=8)
plt.xticks(rotation=15, ha='right', fontsize=9)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()

# Google leads R&D and revenue but trails off on growth and stock impact, while OpenAI starts near zero on spending but surges on revenue growth , then all three converge and collapse to near-zero stock impact,

# ============================================================
# GRAPH 7 — Bubble chart: R&D vs Revenue, size=Stock Impact
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

companies = pop['Company'].unique()
colors = ['#2c7bb6', '#f46d43', '#1a9641', '#d7191c',
          '#7b3294', '#008837', '#e66101', '#5e3c99']

for i, company in enumerate(companies):
    subset = pop[pop['Company'] == company]
    sizes = (subset['Stock_Impact_%'].abs() * 40).clip(lower=30)
    ax.scatter(
        subset['R&D_Spending_USD_Mn'],
        subset['AI_Revenue_USD_Mn'],
        s=sizes,
        color=colors[i % len(colors)],
        alpha=0.65,
        edgecolors='white',
        linewidths=0.6,
        label=company,
        zorder=3
    )

# Trend line — fixed: x_line uses R&D min/max not Revenue
add_trendline(ax,
              pop['R&D_Spending_USD_Mn'],
              pop['AI_Revenue_USD_Mn'],
              color='#333333')

ax.set_title('R&D Spending vs AI Revenue\nbubble size = Stock Impact magnitude',
             fontsize=13, fontweight='bold', color='#1a1a1a',
             loc='left', pad=14)
ax.set_xlabel('R&D Spending (USD Mn)', fontsize=11,
              color='#555555', labelpad=10)
ax.set_ylabel('AI Revenue (USD Mn)', fontsize=11,
              color='#555555', labelpad=10)
ax.legend(frameon=False, fontsize=9,
          loc='upper left', bbox_to_anchor=(1, 1))
ax.tick_params(colors='#888888', length=0)
plt.tight_layout()
plt.show()

# Meta beats the trend line with bigger stock impact bubbles, Google follows it closely, and OpenAI's oversized bubbles near zero prove its stock runs on hype not fundamentals.
print('---------------------------')


print('*************-------------**************')
