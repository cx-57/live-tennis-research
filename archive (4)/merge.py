import pandas as pd 

# shared unmatched list for all years
unmatched = []

# 2024
print("-------------2024-------------")

d1 = pd.read_csv("2024x.csv")
d2 = pd.read_csv("2024y.csv")

d1 = d1[d1["Comment"] == "Completed"].reset_index(drop=True)

# helper functions 
def normalize_d1_name(name):
    parts = str(name).strip().split()
    return parts[:-1][-1].lower()

def normalize_d2_name(name):
    return str(name).strip().split()[-1].lower()


# to help normalize very different tournament names between the 2 datasets
aliases = {
    "indian wells": "indian wells masters",
    "miami":        "miami masters",
    "monte carlo":  "monte carlo masters",
    "madrid":       "madrid masters",
    "rome":         "rome masters",
    "canada":       "canada masters",
    "toronto":      "canada masters",
    "cincinnati":   "cincinnati masters",
    "shanghai":     "shanghai masters",
    "paris":        "paris masters",
    "melbourne":    "australian open",
    "new york":     "us open",
    "queens club":  "queen's club",
    "london":       "wimbledon",
    "'s-hertogenbosch": "s hertogenbosch",
    "turin": "tour finals",
    "nur-sultan": "astana",
}

def normalize_tourney(name):
    n = str(name).strip().lower()
    return aliases.get(n, n)

d1["la"] = d1["Winner"].apply(normalize_d1_name)
d1["lb"] = d1["Loser"].apply(normalize_d1_name)
d1["t"]  = d1["Location"].apply(normalize_tourney)

d2["la"] = d2["winner_name"].apply(normalize_d2_name)
d2["lb"] = d2["loser_name"].apply(normalize_d2_name)
d2["t"]  = d2["tourney_name"].apply(normalize_tourney)

d2_map = {}
for i, row in d2.iterrows():
    d2_map[(row["t"], row["la"], row["lb"])] = i

skip = {"Winner", "Loser", "Location", "Tournament", "la", "lb", "t"}
d1_cols = [col for col in d1.columns if col not in skip]

for col in d1_cols:
    d2[f"d1_{col}"] = None

for _, d1_row in d1.iterrows():
    key = (d1_row["t"], d1_row["la"], d1_row["lb"])
    if key in d2_map:
        for col in d1_cols:
            d2.at[d2_map[key], f"d1_{col}"] = d1_row[col]
    else:
        unmatched.append(d1_row)

d2.drop(columns=["la", "lb", "t"], inplace=True)
d2.to_csv("merged_2024.csv", index=False)
print(f"Matched: {len(d1) - sum(1 for u in unmatched)} / {len(d1)}")


# 2023 - same process 
print("-------------2023-------------")

d3 = pd.read_csv("2023x.csv")
d4 = pd.read_csv("2023y.csv")

d3 = d3[d3["Comment"] == "Completed"].reset_index(drop=True)

d3["la"] = d3["Winner"].apply(normalize_d1_name)
d3["lb"] = d3["Loser"].apply(normalize_d1_name)
d3["t"]  = d3["Location"].apply(normalize_tourney)

d4["la"] = d4["winner_name"].apply(normalize_d2_name)
d4["lb"] = d4["loser_name"].apply(normalize_d2_name)
d4["t"]  = d4["tourney_name"].apply(normalize_tourney)

d4_map = {}
for i, row in d4.iterrows():
    d4_map[(row["t"], row["la"], row["lb"])] = i

d3_cols = [col for col in d3.columns if col not in skip]

for col in d3_cols:
    d4[f"d1_{col}"] = None

unmatched_2023_start = len(unmatched)
for _, d3_row in d3.iterrows():
    key = (d3_row["t"], d3_row["la"], d3_row["lb"])
    if key in d4_map:
        for col in d3_cols:
            d4.at[d4_map[key], f"d1_{col}"] = d3_row[col]
    else:
        unmatched.append(d3_row)

d4.drop(columns=["la", "lb", "t"], inplace=True)
d4.to_csv("merged_2023.csv", index=False)
print(f"Matched: {len(d3) - (len(unmatched) - unmatched_2023_start)} / {len(d3)}")


# save all unmatched rows to one csv
if unmatched:
    pd.DataFrame(unmatched).to_csv("unmatched.csv", index=False)
    print(f"Unmatched rows saved to unmatched.csv")


# combine everything
merged_2023 = pd.read_csv("merged_2023.csv")
merged_2024 = pd.read_csv("merged_2024.csv")

df = pd.concat([merged_2023, merged_2024], ignore_index=True)
df.to_csv("merged_all.csv", index=False)
print(f"Total rows in merged_all.csv: {len(df)}")