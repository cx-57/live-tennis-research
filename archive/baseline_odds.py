import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("merged_all.csv")

bookmakers = {
    "Bet365":       ("d1_B365W",  "d1_B365L"),
    "Pinnacle":     ("d1_PSW",    "d1_PSL"),
    "Max":          ("d1_MaxW",   "d1_MaxL"),
    "Avg":          ("d1_AvgW",   "d1_AvgL"),
}

results = []

for name, (w_col, l_col) in bookmakers.items():
    if w_col not in df.columns or l_col not in df.columns:
        print(f"{name} — columns not found, skipping")
        continue
    
    temp = df.dropna(subset=[w_col, l_col])
    if len(temp) == 0:
        print(f"{name} — no data, skipping")
        continue

    # lower odds = favourite, predict favourite always wins
    predicted = (temp[w_col] < temp[l_col]).astype(int)
    actual = pd.Series([1] * len(temp))
    acc = accuracy_score(actual, predicted)
    results.append((name, acc, len(temp)))

results.sort(key=lambda x: x[1], reverse=True)
print(f"{'Bookmaker':<15} {'Accuracy':<10} {'Matches'}")
print("-" * 35)
for name, acc, n in results:
    print(f"{name:<15} {acc:.3f}      {n}")