from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import json
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('pot/top_10_ai_stocks.csv', delimiter=',')

print(df['symbol'].unique())

SYMBOLS = ['NVDA', 'META', 'ANET', 'AMZN',
           'PANW', 'NOW', 'AMD', 'PATH', 'TSLA', 'AI']
START = "2018-01-01"
END = "2024-01-01"
SEQ_LEN = 30       # lookback window (days)
BATCH = 512
EPOCHS = 100
TRAIN_SPLIT = 0.80

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────


def build_features(df):
    """
    Input : raw OHLC DataFrame (any column casing, with or without Adj Close)
    Output: (feature_df, log_close series)
    """
    df = df.copy().dropna()
    df.columns = [c.lower().strip() for c in df.columns]

    # Resolve adjusted close
    adj_col = next((c for c in df.columns if "adj" in c), None)

    df["lc"] = np.log(df["close"])
    df["lh"] = np.log(df["high"])
    df["ll"] = np.log(df["low"])
    df["lo"] = np.log(df["open"])
    if adj_col:
        df["ladj"] = np.log(df[adj_col])
    else:
        df["ladj"] = df["lc"]

    # Returns
    df["ret1"] = df["lc"].diff()
    df["ret2"] = df["lc"].diff(2)
    df["ret5"] = df["lc"].diff(5)

    # Moving averages
    df["ma5"] = df["lc"].rolling(5).mean()
    df["ma10"] = df["lc"].rolling(10).mean()
    df["ma20"] = df["lc"].rolling(20).mean()

    # Spread between MAs (momentum)
    df["ma_spread"] = df["ma5"] - df["ma20"]

    # Volatility
    df["vol5"] = df["ret1"].rolling(5).std()
    df["vol10"] = df["ret1"].rolling(10).std()

    # Price position in daily high-low range
    df["hl_pct"] = (df["lc"] - df["ll"]) / (df["lh"] - df["ll"] + 1e-9)

    # RSI (14-period)
    delta = df["lc"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12 = df["lc"].ewm(span=12, adjust=False).mean()
    ema26 = df["lc"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_h"] = df["macd"] - df["sig"]

    # Bollinger Band position
    bb_mid = df["lc"].rolling(20).mean()
    bb_std = df["lc"].rolling(20).std()
    df["bb_pos"] = (df["lc"] - bb_mid) / (bb_std + 1e-9)

    df.dropna(inplace=True)

    feature_cols = [
        "lo", "lh", "ll", "lc", "ladj",
        "ret1", "ret2", "ret5",
        "ma5", "ma10", "ma20", "ma_spread",
        "vol5", "vol10",
        "hl_pct", "rsi", "macd", "sig", "macd_h", "bb_pos",
    ]
    return df[feature_cols], df["lc"]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading data …")
raw = yf.download(SYMBOLS, start=START, end=END,
                  auto_adjust=True, progress=False)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PER-STOCK SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────
per_stock = {}   # sym -> dict of arrays

print("\nBuilding sequences …")
for sym in SYMBOLS:
    try:
        # Extract single-stock DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            df_sym = raw.xs(sym, axis=1, level=1).dropna()
        else:
            df_sym = raw.copy().dropna()

        if len(df_sym) < SEQ_LEN + 50:
            print(f"  {sym}: too few rows, skipping")
            continue

        feats, lc = build_features(df_sym)
        n = len(feats)
        split_idx = int(n * TRAIN_SPLIT)

        # ── Feature scaler: fit on TRAIN rows only ──────────────────
        xs = MinMaxScaler()
        xs.fit(feats.iloc[:split_idx].values)
        X_scaled = xs.transform(feats.values)

        lc_vals = lc.values
        # log returns:  ret[i] = lc[i+1] - lc[i]  →  length n-1
        log_rets = np.diff(lc_vals)

        X_tr, y_tr = [], []
        X_te, y_te, base_te = [], [], []

        for i in range(SEQ_LEN, n - 1):
            seq = X_scaled[i - SEQ_LEN: i]          # shape (SEQ_LEN, n_feat)
            target = log_rets[i]                         # next-day log return
            base = lc_vals[i]                          # log_close at time i

            if i < split_idx:
                X_tr.append(seq)
                y_tr.append(target)
            else:
                X_te.append(seq)
                y_te.append(target)
                base_te.append(base)

        per_stock[sym] = {
            "X_train": np.array(X_tr,    dtype=np.float32),
            "X_test": np.array(X_te,    dtype=np.float32),
            "y_train": np.array(y_tr,    dtype=np.float32),
            "y_test": np.array(y_te,    dtype=np.float32),
            "lc_base": np.array(base_te, dtype=np.float32),
            "dates": feats.index,        # full index; trimmed at eval time
            "split_idx": split_idx,
        }
        print(f"  {sym}: {n} rows | train={len(X_tr)} test={len(X_te)} sequences")

    except Exception as e:
        print(f"  {sym}: error — {e}")

if not per_stock:
    raise RuntimeError(
        "No stocks loaded. Check internet connection / symbols.")


# ─────────────────────────────────────────────────────────────────────────────
# POOL & SHUFFLE TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
X_train_all = np.concatenate([v["X_train"] for v in per_stock.values()])
y_train_all = np.concatenate([v["y_train"] for v in per_stock.values()])

perm = np.random.permutation(len(X_train_all))
X_train_all = X_train_all[perm]
y_train_all = y_train_all[perm]

n_feat = X_train_all.shape[2]
print(f"\nTotal train sequences : {len(X_train_all)}")
print(f"Features per step     : {n_feat}")
print(
    f"Target range (returns): {y_train_all.min():.4f} → {y_train_all.max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
model = Sequential([
    Input(shape=(SEQ_LEN, n_feat)),

    GRU(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    GRU(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1),   # predicts next-day log return
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="huber",
)
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1),
]

print("\nTraining …")
history = model.fit(
    X_train_all, y_train_all,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION — per stock
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_symbol(sym):
    d = per_stock[sym]

    # Predict log returns
    pred_ret = model.predict(d["X_test"], verbose=0).flatten()
    true_ret = d["y_test"]
    lc_base = d["lc_base"]

    # Reconstruct prices
    #   price[t+1] = exp( log_close[t] + log_return[t] )
    pred_prices = np.exp(lc_base + pred_ret)
    actual_prices = np.exp(lc_base + true_ret)

    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    mae = mean_absolute_error(actual_prices, pred_prices)
    r2 = r2_score(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices) /
                   (actual_prices + 1e-9))) * 100

    # Directional accuracy on RETURNS (the thing we actually predict)
    dir_acc = np.mean(np.sign(true_ret) == np.sign(pred_ret)) * 100

    naive = actual_prices[:-1]
    theil_u = (np.sqrt(np.mean((actual_prices[1:] - pred_prices[1:]) ** 2)) /
               (np.sqrt(np.mean((actual_prices[1:] - naive) ** 2)) + 1e-9))
    pearson = float(np.corrcoef(actual_prices, pred_prices)[0, 1])
    logcosh = float(np.mean(np.log(np.cosh(pred_prices - actual_prices))))
    # Trim dates to exactly match prediction length (avoids shape mismatch)
    test_dates = d["dates"][-len(actual_prices):]

    return (
        dict(rmse=rmse, mae=mae, r2=r2, mape=mape,
             dir_acc=dir_acc, theil_u=theil_u, pearson=pearson, logcosh=logcosh),
        actual_prices,
        pred_prices,
        test_dates,
    )


print("\n" + "=" * 55)
print("PER-SYMBOL TEST RESULTS")
print("=" * 55)

all_metrics = {}
all_actuals = {}
all_preds = {}
all_dates = {}

for sym in per_stock:
    metrics, y_actual, predictions, dates = evaluate_symbol(sym)
    all_metrics[sym] = metrics
    all_actuals[sym] = y_actual
    all_preds[sym] = predictions
    all_dates[sym] = dates

    print(f"\n  {sym}")
    print(f"    RMSE    : ${metrics['rmse']:.4f}")
    print(f"    MAE     : ${metrics['mae']:.4f}")
    print(f"    R²      : {metrics['r2']:.4f}")
    print(f"    MAPE    : {metrics['mape']:.2f}%")
    print(f"    Dir Acc : {metrics['dir_acc']:.2f}%")
    print(f"    Theil U : {metrics['theil_u']:.4f}")
    print(f"    Pred $  : ${predictions.min():.2f} – ${predictions.max():.2f}")
    print(f"    Real $  : ${y_actual.min():.2f} – ${y_actual.max():.2f}")
    print(f"    Pearson : {metrics['pearson']:.4f}")
    print(f"    LogCosh : {metrics['logcosh']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("AGGREGATE METRICS")
print("=" * 55)
for key in ["rmse", "mae", "r2", "mape", "dir_acc", "theil_u", "pearson", "logcosh"]:
    vals = [m[key] for m in all_metrics.values()]
    suffix = "%" if key in ("mape", "dir_acc") else ""
    print(
        f"  Avg {key.upper():8s}: {np.mean(vals):.4f}{suffix}  (std {np.std(vals):.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
n_syms = len(all_actuals)
fig, axes = plt.subplots(n_syms, 1, figsize=(14, 4 * n_syms),
                         constrained_layout=True)
if n_syms == 1:
    axes = [axes]

for ax, sym in zip(axes, all_actuals):
    m = all_metrics[sym]
    ax.plot(all_dates[sym], all_actuals[sym],
            label="Actual", linewidth=1.5, color="steelblue")
    ax.plot(all_dates[sym], all_preds[sym],
            label="Predicted", linewidth=1.2,
            linestyle="--", alpha=0.85, color="coral")
    ax.set_title(
        f"{sym}  |  R²={m['r2']:.3f}   RMSE=${m['rmse']:.2f}   "
        f"MAPE={m['mape']:.2f}%   DirAcc={m['dir_acc']:.1f}%",
        fontsize=11,
    )
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")
plt.suptitle("Stock Price Predictions — Test Set", fontsize=14)
plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPredictions plot saved → predictions.png")

# Loss curve
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(history.history["loss"],     label="Train Loss", linewidth=1.5)
ax2.plot(history.history["val_loss"], label="Val Loss",   linewidth=1.5)
ax2.set_title("Training History (Huber Loss on Log Returns)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Loss curve saved → training_loss.png")

print('---------------------------------')
# mean of all metrics

avg_rmse = np.mean([m["rmse"] for m in all_metrics.values()])
avg_mae = np.mean([m["mae"] for m in all_metrics.values()])
avg_r2 = np.mean([m["r2"] for m in all_metrics.values()])
avg_mape = np.mean([m["mape"] for m in all_metrics.values()])
avg_dir_acc = np.mean([m["dir_acc"] for m in all_metrics.values()])
avg_pearson = np.mean([m["pearson"] for m in all_metrics.values()])
avg_cos_h = np.mean([m["logcosh"] for m in all_metrics.values()])

print("\n" + "=" * 40)
print("      AVERAGE MODEL PERFORMANCE")
print("=" * 40)
print(f"Average RMSE     : {avg_rmse:.2f}")
print(f"Average MAE      : {avg_mae:.2f}")
print(f"Average R²       : {avg_r2:.4f}")
print(f"Average MAPE     : {avg_mape:.2f}%")
print(f"Average Dir Acc  : {avg_dir_acc:.2f}%")
print(f"Average pearson  : {avg_pearson:.2f}%")
print(f"Average logchosh  : {avg_cos_h:.2f}%")

print("=" * 40)

print(
    f"R² Std Dev       : {np.std([m['r2'] for m in all_metrics.values()]):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE METRICS TO JSON
# ─────────────────────────────────────────────────────────────────────────────
def np_converter(obj):
    """Convert numpy types to native Python so json.dump works."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


# Save per-symbol metrics in flat JSON format for cross-model comparison.
output = {
    sym: {k: float(v) for k, v in m.items()}
    for sym, m in all_metrics.items()
}

with open('gru_metrics.json', 'w') as f:
    json.dump(output, f, indent=2, default=np_converter)

print("Saved -> gru_metrics.json\n")

# Note: Model trained on historical stock data; past performance does not guarantee future results.
