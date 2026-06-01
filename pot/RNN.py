from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
#  LOAD & SORT DATA
# ══════════════════════════════════════════════════════════════════
data = pd.read_csv('pot/top_10_ai_stocks.csv', delimiter=',')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.sort_index(inplace=True)

symbols = data['symbol'].unique()
LOOKBACK = 20
features = ['open', 'high', 'low', 'volume', 'adjusted']
target = 'adjusted'
N_FEATURES = len(features)

print(f"Found {len(symbols)} symbols: {list(symbols)}\n")

# ══════════════════════════════════════════════════════════════════
#  BUILD ONE COMBINED DATASET FROM ALL 10 STOCKS
# ══════════════════════════════════════════════════════════════════
all_X, all_y = [], []

# keep per-stock scalers + raw arrays for prediction later
stock_meta = {}

for symbol in symbols:
    sd = data[data['symbol'] == symbol].copy().dropna()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(sd[features])
    y_scaled = y_scaler.fit_transform(sd[[target]])

    for i in range(LOOKBACK, len(X_scaled)):
        all_X.append(X_scaled[i - LOOKBACK:i])
        all_y.append(y_scaled[i])

    stock_meta[symbol] = {
        'sd': sd,
        'X_scaled': X_scaled,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
    }

X = np.array(all_X)
y = np.array(all_y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(
    f"Combined dataset — Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

# ══════════════════════════════════════════════════════════════════
#  BUILD & TRAIN ONE SHARED MODEL
# ══════════════════════════════════════════════════════════════════
inputs = keras.Input(shape=(LOOKBACK, N_FEATURES))
x = keras.layers.GaussianNoise(0.01)(inputs)
x = keras.layers.Bidirectional(
    keras.layers.SimpleRNN(128, return_sequences=True))(x)
x = keras.layers.LayerNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.SimpleRNN(64, return_sequences=True)(x)
x = keras.layers.LayerNormalization()(x)
x = keras.layers.Dropout(0.2)(x)
attn = keras.layers.Attention()([x, x])
x = keras.layers.Add()([x, attn])
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dense(64, activation='swish')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(32, activation='swish')(x)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss='huber', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ],
    verbose=1
)

# ══════════════════════════════════════════════════════════════════
#  PER-STOCK PREDICTIONS & METRICS  (model is already trained)
# ══════════════════════════════════════════════════════════════════
all_metrics = {}
all_actuals = {}
all_preds = {}
all_dates = {}

for symbol, meta in stock_meta.items():
    sd = meta['sd']
    X_scaled = meta['X_scaled']
    y_scaler = meta['y_scaler']

    # rebuild sequences for this stock only
    Xs, ys = [], []
    for i in range(LOOKBACK, len(X_scaled)):
        Xs.append(X_scaled[i - LOOKBACK:i])
        ys.append(y_scaler.transform(sd[[target]].values[i:i+1]))
    Xs = np.array(Xs)
    ys = np.array(ys)

    split_s = int(len(Xs) * 0.8)
    Xs_test = Xs[split_s:]
    ys_test = ys[split_s:]

    predictions = y_scaler.inverse_transform(
        model.predict(Xs_test, verbose=0)).flatten()
    y_test_actual = y_scaler.inverse_transform(
        ys_test.reshape(-1, 1)).flatten()
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

    last_seq = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, N_FEATURES)
    next_day_price = float(y_scaler.inverse_transform(
        model.predict(last_seq, verbose=0))[0][0])
    last_price = float(sd['adjusted'].iloc[-1])
    last_date = sd.index[-1]
    next_date = last_date + pd.tseries.offsets.BDay(1)

    all_metrics[symbol] = {
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "dir_acc": dir_acc, "theil_u": theil_u, "logcosh": logcosh, "pearson": pearson,
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
print("\n" + "═" * 84)
print(f"  {'SYM':<6} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'DirAcc':>8} {'TheilU':>8} {'Pearson':>8}")
print("═" * 84)
for sym, m in all_metrics.items():
    flag = " ✓" if m['dir_acc'] >= 55 else ""
    print(f"  {sym:<6} ${m['rmse']:>7.2f} ${m['mae']:>7.2f} {m['r2']:>8.4f} "
          f"{m['mape']:>7.2f}% {m['dir_acc']:>7.2f}%{flag}  {m['theil_u']:>7.4f}  {m['pearson']:>7.4f}")
print("═" * 84)
print(
    f"  {'Average DirAcc':<30}: {np.mean([m['dir_acc'] for m in all_metrics.values()]):.2f}%")
print("═" * 84)

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

# Plot 1 — Actual vs Predicted + Loss curve
fig, axes = plt.subplots(len(symbols), 2, figsize=(18, 5 * len(symbols)))
for idx, sym in enumerate(symbols):
    axes[idx, 0].plot(all_dates[sym], all_actuals[sym],
                      label='Actual', color='steelblue', linewidth=1.5)
    axes[idx, 0].plot(all_dates[sym], all_preds[sym],   label='Predicted',
                      color='orangered', linewidth=1.5, linestyle='--')
    axes[idx, 0].set_title(f'{sym} — Actual vs Predicted (test set)')
    axes[idx, 0].set_xlabel('Date')
    axes[idx, 0].set_ylabel('Price (USD)')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)

    axes[idx, 1].plot(history.history['loss'],
                      label='Train Loss', color='steelblue')
    axes[idx, 1].plot(history.history['val_loss'],
                      label='Val Loss',   color='orangered')
    axes[idx, 1].set_title(f'{sym} — Shared Model Loss')
    axes[idx, 1].set_xlabel('Epoch')
    axes[idx, 1].set_ylabel('Loss (Huber)')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()
print("Saved → rnn_all_stocks.png")

# Plot 2 — Summary bar chart
syms = list(symbols)
dir_accs = [all_metrics[s]['dir_acc'] for s in syms]
r2s = [max(0, all_metrics[s]['r2']) for s in syms]

BG, ACCENT, PRED_C, GRID_C = '#0d0d0d', '#00FFAA', '#FF6B6B', '#2a2a2a'
plt.style.use('dark_background')
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
bar_colors = [ACCENT if d >= 50 else PRED_C for d in dir_accs]

for ax in ax2:
    ax.set_facecolor('#111111')
    ax.grid(True, color=GRID_C, linewidth=0.5, axis='y')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

ax2[0].bar(syms, dir_accs, color=bar_colors,
           edgecolor='#333333', linewidth=0.8)
ax2[0].axhline(50, color='white', linewidth=1, linestyle=':',
               alpha=0.5, label='Random baseline (50%)')
ax2[0].set_title("Directional Accuracy by Stock",
                 color='white', fontsize=12, fontweight='bold')
ax2[0].set_ylabel("Accuracy (%)", color='#aaaaaa')
ax2[0].set_ylim(0, 100)
ax2[0].legend(fontsize=9, framealpha=0.2, labelcolor='white')
for i, v in enumerate(dir_accs):
    ax2[0].text(i, v + 1, f"{v:.1f}%", ha='center',
                color='white', fontsize=8, fontweight='bold')

ax2[1].bar(syms, r2s, color=ACCENT, edgecolor='#333333', linewidth=0.8)
ax2[1].set_title("R² Score by Stock (clipped at 0)",
                 color='white', fontsize=12, fontweight='bold')
ax2[1].set_ylabel("R²", color='#aaaaaa')
ax2[1].set_ylim(0, 1)
for i, v in enumerate(r2s):
    ax2[1].text(i, v + 0.01, f"{v:.3f}", ha='center',
                color='white', fontsize=8, fontweight='bold')

fig2.suptitle("Per-Stock Performance Summary",
              color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("metrics_summary.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved → metrics_summary.png")

# ══════════════════════════════════════════════════════════════════
#  SAVE JSON
# ══════════════════════════════════════════════════════════════════
with open('rnn_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("Saved → rnn_metrics.json\n")

with open('rnn_metrics.json', 'r') as f:
    for sym, m in json.load(f).items():
        print(f"{sym}: RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | "
              f"R2={m['r2']:.4f} | DirAcc={m['dir_acc']:.1f}% | "
              f"Next→${m['next_day']['predicted_price']:.2f}")

# Note: Model trained on historical stock data; past performance does not guarantee future results.
