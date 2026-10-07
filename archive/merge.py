import pandas as pd

unmatched = []

aliases = {
    "indian wells": "indian wells masters",
    "miami": "miami masters",
    "monte carlo": "monte carlo masters",
    "madrid": "madrid masters",
    "rome": "rome masters",
    "canada": "canada masters",
    "toronto": "canada masters",
    "cincinnati": "cincinnati masters",
    "shanghai": "shanghai masters",
    "paris": "paris masters",
    "melbourne": "australian open",
    "new york": "us open",
    "queens club": "queen's club",
    "london": "wimbledon",
    "'s-hertogenbosch": "s hertogenbosch",
    "turin": "tour finals",
    "nur-sultan": "astana",
}

def normalize_d1_name(name):
    parts = str(name).strip().split()
    return parts[-2].lower() if len(parts) > 1 else str(name).lower()

def normalize_d2_name(name):
    return str(name).strip().split()[-1].lower()

def normalize_tourney(name):
    n = str(name).strip().lower()
    return aliases.get(n, n)


def merge_year(year):
    print(f"-------------{year}-------------")

    d1 = pd.read_csv(f"{year}x.csv")
    d2 = pd.read_csv(f"{year}y.csv")

    d1 = d1[d1["Comment"] == "Completed"].reset_index(drop=True)

    # normalize
    d1["la"] = d1["Winner"].apply(normalize_d1_name)
    d1["lb"] = d1["Loser"].apply(normalize_d1_name)
    d1["t"]  = d1["Location"].apply(normalize_tourney)

    d2["la"] = d2["winner_name"].apply(normalize_d2_name)
    d2["lb"] = d2["loser_name"].apply(normalize_d2_name)
    d2["t"]  = d2["tourney_name"].apply(normalize_tourney)

    # map
    d2_map = {
        (row["t"], row["la"], row["lb"]): i
        for i, row in d2.iterrows()
    }

    skip = {"Winner", "Loser", "Location", "Tournament", "la", "lb", "t"}
    d1_cols = [c for c in d1.columns if c not in skip]

    for col in d1_cols:
        d2[f"d1_{col}"] = None

    local_unmatched = []

    for _, r in d1.iterrows():
        key = (r["t"], r["la"], r["lb"])
        if key in d2_map:
            idx = d2_map[key]
            for col in d1_cols:
                d2.at[idx, f"d1_{col}"] = r[col]
        else:
            local_unmatched.append(r)

    d2.drop(columns=["la", "lb", "t"], inplace=True)
    out_file = f"merged_{year}.csv"
    d2.to_csv(out_file, index=False)

    unmatched.extend(local_unmatched)

    print(f"Matched: {len(d1) - len(local_unmatched)} / {len(d1)}")
    return out_file


# run all years
years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
outputs = [merge_year(y) for y in years]


# combine everything
df = pd.concat([pd.read_csv(f) for f in outputs], ignore_index=True)
df.to_csv("merged_all.csv", index=False)
print(f"Total number of rows: {len(df)}")

# save unmatched
if unmatched:
    pd.DataFrame(unmatched).to_csv("unmatched.csv", index=False)
    print(f"Unmatched rows saved: {len(unmatched)}")