import pandas as pd 


d1 = pd.read_csv("2024x.csv")
d2 = pd.read_csv("2024y.csv")


# remove Walkovers, retirements, and other non-completed matches
d1 = d1[d1["Comment"] == "Completed"].reset_index(drop=True)


# helper functions 
def normalize_d1_name(name):
    parts = str(name).strip().split()
    return parts[:-1][-1].lower()

def normalize_d2_name(name):
    return str(name).strip().split()[-1].lower()

aliases = {
    "indian wells": "indian wells masters",
    "miami":        "miami masters",
    "monte carlo":  "monte carlo masters",
    "madrid":       "madrid masters",
    "rome":         "rome masters",
    "canada":       "canada masters",
    "cincinnati":   "cincinnati masters",
    "shanghai":     "shanghai masters",
    "paris":        "paris masters",
    "melbourne":    "australian open",
    "new york":     "us open",
    "queens club":  "queen's club",
    "london":       "wimbledon",
    "'s-hertogenbosch": "s hertogenbosch",

}

def normalize_tourney(name):
    n = str(name).strip().lower()
    return aliases.get(n, n)


# adding columns to both d1 and d2 
d1["la"] = d1["Winner"].apply(normalize_d1_name)
d1["lb"] = d1["Loser"].apply(normalize_d1_name)
d1["t"] = d1["Location"].apply(normalize_tourney)

d2["la"] = d2["winner_name"].apply(normalize_d2_name)
d2["lb"] = d2["loser_name"].apply(normalize_d2_name)
d2["t"] = d2["tourney_name"].apply(normalize_tourney)


# build a hash map for d2
d2_map = {}
for i, row in d2.iterrows():
    key1 = (row["t"], row["la"], row["lb"])
    d2_map[key1] = i


# create columns for x (skip the columns that were just created)
skip = {"Winner", "Loser", "Location", "Tournament", "la", "lb", "t"}
d1_cols = []
for col in d1.columns:
    if col not in skip:
        d1_cols.append(col)


# add the same amount of empty columns to d2 as the amount of data in d1
for col in d1_cols:
    d2[f"d1_{col}"] = None


# Merge unique d1 columns into d2
unmatched = []
for _, d1_row in d1.iterrows():
    key = (d1_row["t"], d1_row["la"], d1_row["lb"])
    if key in d2_map:
        for col in d1_cols:
            d2.at[d2_map[key], f"d1_{col}"] = d1_row[col]
    else:
        unmatched.append(d1_row)


# clean up everything and save merged.csv and unmatched.csv
d2.drop(columns=["la", "lb", "t"], inplace=True)
d2.to_csv("merged.csv", index=False)

print(f"Matched: {len(d1) - len(unmatched)} / {len(d1)}")
if unmatched:
    pd.DataFrame(unmatched).to_csv("unmatched.csv", index=False)
    print(f"Unmatched rows saved to unmatched.csv")
