import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Dropout,
                                     Bidirectional, LSTM, Dense)
from tensorflow.keras.models import Sequential
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

tf.random.set_seed(42)
np.random.seed(42)

# it helpens faster the running process else it's like turtle running a marathon
print(tf.config.list_physical_devices('GPU'))

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
data = pd.read_csv('pot/top_10_ai_stocks.csv', delimiter=',')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.sort_index(inplace=True)

symbols = data['symbol'].unique()
LOOKBACK = 30
features = ['open', 'high', 'low', 'close', 'adjusted']
target = 'adjusted'
N_FEATURES = len(features)

print(f"Found {len(symbols)} symbols: {list(symbols)}\n")

# ══════════════════════════════════════════════════════════════════
#  PHASE 1 — BUILD ONE COMBINED DATASET FROM ALL 10 STOCKS
# ══════════════════════════════════════════════════════════════════
all_X, all_y = [], []
stock_meta = {}

for symbol in symbols:
    sd = data[data['symbol'] == symbol].copy().dropna()
    train_end = int(len(sd) * 0.8)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(sd[features].iloc[:train_end])
    y_scaler.fit(sd[[target]].iloc[:train_end])

    X_scaled = x_scaler.transform(sd[features])
    y_scaled = y_scaler.transform(sd[[target]])

    for i in range(LOOKBACK, len(X_scaled)):
        all_X.append(X_scaled[i - LOOKBACK:i])
        all_y.append(y_scaled[i])

    stock_meta[symbol] = {'sd': sd, 'X_scaled': X_scaled, 'y_scaler': y_scaler}

X = np.array(all_X)
y = np.array(all_y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(
    f"Combined dataset — Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

# ══════════════════════════════════════════════════════════════════
#  PHASE 2 — ONE SHARED Sequential MODEL
# ══════════════════════════════════════════════════════════════════
model = Sequential([
    Input(shape=(LOOKBACK, N_FEATURES)),

    Conv1D(64, kernel_size=3, activation='relu', padding='causal'),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(32, kernel_size=5, activation='relu', padding='causal'),
    BatchNormalization(),
    Dropout(0.2),

    Bidirectional(LSTM(128, return_sequences=True,
                  kernel_regularizer=keras.regularizers.l2(0.001))),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=True,
                  kernel_regularizer=keras.regularizers.l2(0.001))),
    Dropout(0.2),

    LSTM(32, return_sequences=False),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=512,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ],
    verbose=1
)

# ══════════════════════════════════════════════════════════════════
#  PHASE 3 — PER-STOCK PREDICTIONS & METRICS
# ══════════════════════════════════════════════════════════════════
all_metrics = {}
all_actuals = {}
all_preds = {}
all_dates = {}

for symbol, meta in stock_meta.items():
    sd, X_scaled, y_scaler = meta['sd'], meta['X_scaled'], meta['y_scaler']

    Xs, ys = [], []
    for i in range(LOOKBACK, len(X_scaled)):
        Xs.append(X_scaled[i - LOOKBACK:i])
        ys.append(y_scaler.transform(sd[[target]].values[i:i+1]))
    Xs = np.array(Xs)
    ys = np.array(ys)

    split_s = int(len(Xs) * 0.8)
    predictions = y_scaler.inverse_transform(
        model.predict(Xs[split_s:], verbose=0)).flatten()
    y_test_actual = y_scaler.inverse_transform(
        ys[split_s:].reshape(-1, 1)).flatten()
    dates_test = sd.index[LOOKBACK +
                          split_s: LOOKBACK + split_s + len(predictions)]

    rmse = float(np.sqrt(mean_squared_error(y_test_actual, predictions)))
    mae = float(mean_absolute_error(y_test_actual, predictions))
    r2 = float(r2_score(y_test_actual, predictions))
    mape = float(
        np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100)
    dir_acc = float(np.mean(np.sign(np.diff(y_test_actual))
                    == np.sign(np.diff(predictions))) * 100)
    naive = y_test_actual[:-1]
    theil_u = float(np.sqrt(np.mean((y_test_actual[1:] - predictions[1:]) ** 2)) /
                    np.sqrt(np.mean((y_test_actual[1:] - naive) ** 2)))
    logcosh = float(np.mean(np.log(np.cosh(predictions - y_test_actual))))
    pearson = float(np.corrcoef(y_test_actual, predictions)[0, 1])
    msle = float(np.mean((np.log1p(np.maximum(y_test_actual, 0)) -
                          np.log1p(np.maximum(predictions, 0))) ** 2))

    last_seq = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, N_FEATURES)
    next_day_price = float(y_scaler.inverse_transform(
        model.predict(last_seq, verbose=0))[0][0])
    last_price = float(sd['adjusted'].iloc[-1])
    last_date = sd.index[-1]
    next_date = last_date + pd.tseries.offsets.BDay(1)

    all_metrics[symbol] = {
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "dir_acc": dir_acc, "theil_u": theil_u,
        "logcosh": logcosh, "pearson": pearson, "msle": msle,
        "next_day": {
            "last_date": str(last_date.date()), "predict_date": str(next_date.date()),
            "last_price": last_price, "predicted_price": next_day_price,
            "expected_move": next_day_price - last_price,
        }
    }
    all_actuals[symbol] = y_test_actual
    all_preds[symbol] = predictions
    all_dates[symbol] = dates_test

# ══════════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "═" * 90)
print(f"  {'SYM':<6} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'DirAcc':>8} {'TheilU':>8} {'Pearson':>8}")
print("═" * 90)
for sym, m in all_metrics.items():
    flag = " ✓" if m['dir_acc'] >= 55 else ""
    print(f"  {sym:<6} ${m['rmse']:>7.2f} ${m['mae']:>7.2f} {m['r2']:>8.4f} "
          f"{m['mape']:>7.2f}% {m['dir_acc']:>7.2f}%{flag}  {m['theil_u']:>7.4f}  {m['pearson']:>7.4f}")
print("═" * 90)
print(
    f"  {'Average DirAcc':<30}: {np.mean([m['dir_acc'] for m in all_metrics.values()]):.2f}%")
print("═" * 90)

print("\n" + "═" * 62)
print("          NEXT DAY PREDICTIONS — ALL STOCKS")
print("═" * 62)
for sym, m in all_metrics.items():
    nd = m['next_day']
    move = nd['expected_move']
    pct = move / nd['last_price'] * 100
    arrow = '▲' if move > 0 else '▼'
    print(f"  {sym:<6}  {nd['last_date']} → {nd['predict_date']}  "
          f"close=${nd['last_price']:>8.2f}   pred=${nd['predicted_price']:>8.2f}   "
          f"{arrow} {abs(pct):>5.2f}%")
print("═" * 62)

# ══════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════
print("\nSaving plots ...")
BG, ACCENT, PRED_C, GRID_C = '#0d0d0d', '#00FFAA', '#FF6B6B', '#2a2a2a'
plt.style.use('dark_background')

# Plot 1 — Actual vs Predicted
n_syms = len(symbols)
nrows = (n_syms + 1) // 2
fig, axes = plt.subplots(nrows, 2, figsize=(18, 5 * nrows), facecolor=BG)
axes = axes.flatten()

for idx, sym in enumerate(symbols):
    ax = axes[idx]
    ax.set_facecolor('#111111')
    m = all_metrics[sym]
    ax.plot(all_dates[sym], all_actuals[sym], color=ACCENT,
            linewidth=1.8, label='Actual',    alpha=0.9)
    ax.plot(all_dates[sym], all_preds[sym],   color=PRED_C,
            linewidth=1.4, label='Predicted', linestyle='--', alpha=0.85)
    ax.fill_between(all_dates[sym], all_actuals[sym],
                    all_preds[sym], alpha=0.07, color=PRED_C)
    ax.set_title(f"{sym}   R²={m['r2']:.3f}   DirAcc={m['dir_acc']:.1f}%",
                 color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel("Price ($)", color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#666666', labelsize=8)
    ax.grid(True, color=GRID_C, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')

for idx in range(n_syms, len(axes)):
    axes[idx].set_visible(False)
fig.suptitle("LSTM — Actual vs Predicted Price", color='white',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
plt.savefig('lstm_all_stocks.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved → lstm_all_stocks.png")

# Plot 2 — Training curves
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
for ax in ax2:
    ax.set_facecolor('#111111')
    ax.grid(True, color=GRID_C, linewidth=0.5)
    ax.tick_params(colors='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

ax2[0].plot(history.history['loss'],     color=ACCENT,
            linewidth=2, label='Train Loss')
ax2[0].plot(history.history['val_loss'], color=PRED_C,
            linewidth=2, linestyle='--', label='Val Loss')
ax2[0].set_title("Training Loss (MSE)", color='white',
                 fontsize=12, fontweight='bold')
ax2[0].set_xlabel("Epoch", color='#aaaaaa')
ax2[0].set_ylabel("Loss", color='#aaaaaa')
ax2[0].legend(fontsize=10, framealpha=0.2, labelcolor='white')

ax2[1].plot(history.history['mae'],     color=ACCENT,
            linewidth=2, label='Train MAE')
ax2[1].plot(history.history['val_mae'], color=PRED_C,
            linewidth=2, linestyle='--', label='Val MAE')
ax2[1].set_title("Training MAE", color='white', fontsize=12, fontweight='bold')
ax2[1].set_xlabel("Epoch", color='#aaaaaa')
ax2[1].set_ylabel("MAE", color='#aaaaaa')
ax2[1].legend(fontsize=10, framealpha=0.2, labelcolor='white')

fig2.suptitle("LSTM Training Curves", color='white',
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lstm_training_curves.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved → lstm_training_curves.png")

# Plot 3 — Summary bar chart
syms = list(symbols)
dir_accs = [all_metrics[s]['dir_acc'] for s in syms]
r2s = [max(0, all_metrics[s]['r2']) for s in syms]

fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
bar_colors = [ACCENT if d >= 50 else PRED_C for d in dir_accs]
for ax in ax3:
    ax.set_facecolor('#111111')
    ax.grid(True, color=GRID_C, linewidth=0.5, axis='y')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

ax3[0].bar(syms, dir_accs, color=bar_colors,
           edgecolor='#333333', linewidth=0.8)
ax3[0].axhline(50, color='white', linewidth=1, linestyle=':',
               alpha=0.5, label='Random baseline (50%)')
ax3[0].set_title("Directional Accuracy by Stock",
                 color='white', fontsize=12, fontweight='bold')
ax3[0].set_ylabel("Accuracy (%)", color='#aaaaaa')
ax3[0].set_ylim(0, 100)
ax3[0].legend(fontsize=9, framealpha=0.2, labelcolor='white')
for i, v in enumerate(dir_accs):
    ax3[0].text(i, v + 1, f"{v:.1f}%", ha='center',
                color='white', fontsize=8, fontweight='bold')

ax3[1].bar(syms, r2s, color=ACCENT, edgecolor='#333333', linewidth=0.8)
ax3[1].set_title("R² Score by Stock (clipped at 0)",
                 color='white', fontsize=12, fontweight='bold')
ax3[1].set_ylabel("R²", color='#aaaaaa')
ax3[1].set_ylim(0, 1)
for i, v in enumerate(r2s):
    ax3[1].text(i, v + 0.01, f"{v:.3f}", ha='center',
                color='white', fontsize=8, fontweight='bold')

fig3.suptitle("LSTM — Per-Stock Performance Summary",
              color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
plt.close()
print("Saved → lstm_metrics_summary.png")

# ══════════════════════════════════════════════════════════════════
#  SAVE JSON
# ══════════════════════════════════════════════════════════════════
with open('lstm_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("Saved → lstm_metrics.json\n")

with open('lstm_metrics.json', 'r') as f:
    for sym, m in json.load(f).items():
        print(f"{sym}: RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | "
              f"R2={m['r2']:.4f} | DirAcc={m['dir_acc']:.1f}% | "
              f"Next→${m['next_day']['predicted_price']:.2f}")

print('---------------------------------')
# mean of all metrics

avg_rmse = np.mean([m["rmse"] for m in all_metrics.values()])
avg_mae = np.mean([m["mae"] for m in all_metrics.values()])
avg_r2 = np.mean([m["r2"] for m in all_metrics.values()])
avg_mape = np.mean([m["mape"] for m in all_metrics.values()])
avg_dir_acc = np.mean([m["dir_acc"] for m in all_metrics.values()])

print("\n" + "=" * 40)
print("      AVERAGE MODEL PERFORMANCE")
print("=" * 40)
print(f"Average RMSE     : {avg_rmse:.2f}")
print(f"Average MAE      : {avg_mae:.2f}")
print(f"Average R²       : {avg_r2:.4f}")
print(f"Average MAPE     : {avg_mape:.2f}%")
print(f"Average Dir Acc  : {avg_dir_acc:.2f}%")
print("=" * 40)

print(
    f"R² Std Dev       : {np.std([m['r2'] for m in all_metrics.values()]):.4f}")

# Note: Model trained on historical stock data; past performance does not guarantee future results.
