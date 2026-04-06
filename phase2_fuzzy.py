# ============================================================
# PHASE 2 - PLAIN FUZZY SUGENO BASELINE (EXPANDED TO 5 MFs)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

print("="*55)
print("  PHASE 2 - PLAIN FUZZY SUGENO BASELINE")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
X_test = np.load('models/X_test.npy')
y_test = np.load('models/y_test.npy')

print(f"\nTest samples loaded : {X_test.shape[0]}")
print(f"Classes in test set : {np.unique(y_test)}")

# ============================================================
# MEMBERSHIP FUNCTIONS (5 Levels)
# ============================================================
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum(
        (x - a) / (b - a + 1e-9),
        (c - x) / (c - b + 1e-9)
    ))

def get_mf(x_range):
    vl = trimf(x_range, 0,    0,    0.25)
    l  = trimf(x_range, 0,    0.25, 0.5)
    m  = trimf(x_range, 0.25, 0.5,  0.75)
    h  = trimf(x_range, 0.5,  0.75, 1.0)
    vh = trimf(x_range, 0.75, 1.0,  1.0)
    return vl, l, m, h, vh

# ============================================================
# FUZZY RULES (AUTO-GENERATED 125 RULES)
# ============================================================
# Indices: 0:VL, 1:L, 2:M, 3:H, 4:VH
fuzzy_rules = {}

for t in range(5):
    for r in range(5):
        for w in range(5):
            # Logic: Temp increases risk, Wind increases risk, RH decreases risk.
            # Max possible sum is (4 + 4 + 4) = 12
            linear_sum = (t + w + (4 - r)) / 12.0 
            
            # Stretch the mapping symmetrically so it reaches the extremes (0 and 1) easier
            risk_score = (linear_sum * 1.5) - 0.25
            
            # Clip between 0.0 and 1.0 mathematically
            risk_score = max(0.0, min(1.0, risk_score))
            
            fuzzy_rules[(t, r, w)] = risk_score

set_names = ['VL', 'L', 'Med', 'H', 'VH']
print(f"\nFuzzy Rules defined : {len(fuzzy_rules)} rules (A dense 5x5x5 Grid)")
print(f"\n{'Rule':<5} {'Temp':>6} {'RH':>8} {'Wind':>8} {'Output Target':>15}")
print("-" * 50)
for i, ((t, r, w), out) in enumerate(list(fuzzy_rules.items())[:10]):
    print(f"{i+1:<5} {set_names[t]:>6} {set_names[r]:>8} {set_names[w]:>8} {out:>15.3f}")
print("... (showing 10 of 125 rules)")

# ============================================================
# INFERENCE ENGINE
# ============================================================
def fuzzy_sugeno_predict(sample):
    temp_val = sample[4]
    rh_val   = sample[5]
    wind_val = sample[6]

    sets = {}
    for val, name in zip([temp_val, rh_val, wind_val], ['temp', 'rh', 'wind']):
        r = np.array([val])
        vl, l, m, h, vh = get_mf(r)
        sets[name] = [vl[0], l[0], m[0], h[0], vh[0]]

    numerator   = 0
    denominator = 0
    for (ti, ri, wi), output in fuzzy_rules.items():
        firing       = sets['temp'][ti] * sets['rh'][ri] * sets['wind'][wi]
        numerator   += firing * output
        denominator += firing

    if denominator < 1e-9:
        return 0.3
    return numerator / denominator

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
fuzzy_raw = np.array([fuzzy_sugeno_predict(x) for x in X_test])

# Sugeno weighted averages inherently regress to the mean because adjacent overlapping rules fire together.
# We Min-Max scale the outputs to stretch the predictions across the full [0.0 to 1.0] spectrum dynamically.
f_min = fuzzy_raw.min()
f_max = fuzzy_raw.max()
if f_max > f_min:
    fuzzy_raw_scaled = (fuzzy_raw - f_min) / (f_max - f_min)
else:
    fuzzy_raw_scaled = fuzzy_raw

fuzzy_classes = np.round(fuzzy_raw_scaled * 4).astype(int)

acc_fuzzy = accuracy_score(y_test, fuzzy_classes)
f1_fuzzy  = f1_score(y_test, fuzzy_classes,
                     average='weighted', zero_division=0)

print("\n" + "="*40)
print("  Plain Fuzzy Sugeno Results")
print("="*40)
print(f"  Accuracy : {acc_fuzzy*100:.2f}%")
print(f"  F1 Score : {f1_fuzzy:.4f}")
print("="*40)

# Dynamic class names based on what exists in data
labels_map = {0: 'Very Low (No Fire)', 1: 'Low Risk', 2: 'Moderate Risk', 3: 'High Risk', 4: 'Very High Danger'}
unique_classes = np.unique(np.concatenate([y_test, fuzzy_classes]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nDetailed Classification Report:")
print(classification_report(y_test, fuzzy_classes,
      labels=unique_classes,
      target_names=target_names,
      zero_division=0))

# ============================================================
# SAVE RESULTS
# ============================================================
np.save('models/fuzzy_preds.npy', fuzzy_classes)
np.save('models/fuzzy_raw.npy',   fuzzy_raw)
joblib.dump({
    'accuracy':       acc_fuzzy,
    'f1':             f1_fuzzy,
    'unique_classes': unique_classes.tolist()
}, 'models/fuzzy_results.pkl')

print("Results saved to models/ folder.")

# ============================================================
# PLOT 1 - MEMBERSHIP FUNCTIONS
# ============================================================
x_range = np.linspace(0, 1, 200)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
input_labels = ['Temperature', 'Relative Humidity', 'Wind Speed']

for ax, label in zip(axes, input_labels):
    vl, l, m, h, vh = get_mf(x_range)
    ax.plot(x_range, vl, 'purple', label='Very Low', linewidth=2)
    ax.plot(x_range, l,  'blue',   label='Low',      linewidth=2)
    ax.plot(x_range, m,  'green',  label='Medium',   linewidth=2)
    ax.plot(x_range, h,  'orange', label='High',     linewidth=2)
    ax.plot(x_range, vh, 'red',    label='Very High',linewidth=2)
    ax.set_title(f'{label} Membership Functions', fontsize=11)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Membership Degree')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Fuzzy Sugeno — 5 Membership Functions per Variable', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/membership_functions.png', dpi=150)
plt.close()
print("Saved: membership_functions.png")

# ============================================================
# PLOT 2 - PREDICTED VS ACTUAL
# ============================================================
x_pos = np.arange(len(X_test))
plt.figure(figsize=(12, 5))
plt.plot(x_pos, y_test,        'bo', alpha=0.5, markersize=5, label='Actual')
plt.plot(x_pos, fuzzy_classes, 'rx', alpha=0.5, markersize=5, label='Predicted')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Risk Class', fontsize=12)
plt.yticks([0, 1, 2, 3, 4], ['Very Low', 'Low', 'Mod', 'High', 'Very High'])
plt.title('Plain Fuzzy Sugeno — Predicted vs Actual', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_pred_vs_actual.png', dpi=150)
plt.close()
print("Saved: fuzzy_pred_vs_actual.png")

# ============================================================
# PLOT 3 - RAW OUTPUT DISTRIBUTION
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(fuzzy_raw, bins=20, color='tomato', edgecolor='black', alpha=0.85)
plt.xlabel('Raw Fuzzy Output Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Fuzzy Sugeno — Raw Output Distribution', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_output_distribution.png', dpi=150)
plt.close()
print("Saved: fuzzy_output_distribution.png")

print("\n" + "="*55)
print("  PHASE 2 COMPLETE")
print("="*55)