import json
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  LOAD ALL THREE JSON FILES
# ══════════════════════════════════════════════════════════════════
with open('gru_metrics.json',  'r') as f:
    gru_raw = json.load(f)
with open('lstm_metrics.json', 'r') as f:
    lstm_raw = json.load(f)
with open('rnn_metrics.json',  'r') as f:
    rnn_raw = json.load(f)


def flatten(raw):
    """
    If the JSON is nested  {"AMD": {"r2": ..}, "NVDA": {...}}  → average across symbols.
    If it is already flat  {"r2": .., "mae": ..}               → use as-is.
    Normalises key names to lowercase so 'R2' and 'r2' both work.
    """
    # normalise keys to lowercase
    # we added this cause B4 some values were in capital and some small and it was causing error
    # so all we have all three models metrics in lowercase
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()))
        # nested structure — each value is itself a dict of metrics
        if isinstance(first_val, dict) and 'r2' in first_val:
            keys = ['r2', 'mae', 'rmse', 'mape', 'dir_acc',
                    'theil_u', 'logcosh', 'pearson']
            flat = {}
            for k in keys:
                vals = [raw[sym][k] for sym in raw if k in raw[sym]]
                flat[k] = float(np.mean(vals)) if vals else float('nan')
            return flat
        # already flat — just lowercase all keys
        return {k.lower(): v for k, v in raw.items()}
    return raw


gru = flatten(gru_raw)
lstm = flatten(lstm_raw)
rnn = flatten(rnn_raw)

# ══════════════════════════════════════════════════════════════════
#  PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
metrics = [
    ('R²',      'r2',       '.4f'),
    ('MAE',     'mae',      '.4f'),
    ('RMSE',    'rmse',     '.4f'),
    ('MAPE',    'mape',     '.2f'),
    ('DirAcc',  'dir_acc',  '.2f'),
    ('TheilU',  'theil_u',  '.4f'),
    ('LogCosh', 'logcosh',  '.4f'),
    ('Pearson', 'pearson',  '.4f'),
]

print("\n" + "═" * 62)
print(f"  {'METRIC':<10}  {'GRU':>12}  {'LSTM':>12}  {'RNN':>12}")
print("═" * 62)
for label, key, fmt in metrics:
    g = gru.get(key,  float('nan'))
    l = lstm.get(key, float('nan'))
    r = rnn.get(key,  float('nan'))
    print(f"  {label:<10}  {g:>{12}{fmt}}  {l:>{12}{fmt}}  {r:>{12}{fmt}}")
print("═" * 62)

# ══════════════════════════════════════════════════════════════════
#  SCORING  (higher is better)
#  reward high R², DirAcc, Pearson  |  penalise MAE, RMSE, MAPE
# ══════════════════════════════════════════════════════════════════


def score(m):
    reward = m.get('r2', 0) + m.get('dir_acc', 0) / 100 + m.get('pearson', 0)
    penalty = m.get('mae', 0) / 100 + m.get('rmse', 0) / \
        100 + m.get('mape', 0) / 100
    return reward - penalty


scores = {'GRU': score(gru), 'LSTM': score(lstm), 'RNN': score(rnn)}

print(f"\n  Composite scores:")
for name, s in scores.items():
    print(f"    {name}  →  {s:.4f}")

winner = max(scores, key=scores.get)
print(f"\n  🏆  WINNER: {winner}")
print("═" * 62)
