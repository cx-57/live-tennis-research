import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("merged_all.csv")

# drop rows with missing values from just winner and loser rank right now
df = df.dropna(subset = ["winner_rank", "loser_rank"]).reset_index(drop=True)


# randomize data
np.random.seed(42)
flip = np.random.rand(len(df)) > 0.5

df["p1_rank"]        = np.where(flip, df["loser_rank"],        df["winner_rank"])
df["p2_rank"]        = np.where(flip, df["winner_rank"],       df["loser_rank"])
df["p1_rank_points"] = np.where(flip, df["loser_rank_points"], df["winner_rank_points"])
df["p2_rank_points"] = np.where(flip, df["winner_rank_points"],df["loser_rank_points"])
df["p1_age"]         = np.where(flip, df["loser_age"],         df["winner_age"])
df["p2_age"]         = np.where(flip, df["winner_age"],        df["loser_age"])
df["p1_odds"]        = np.where(flip, df["d1_B365L"],          df["d1_B365W"])
df["p2_odds"]        = np.where(flip, df["d1_B365W"],          df["d1_B365L"])
df["p1_odds_psw"]    = np.where(flip, df["d1_PSW"],            df["d1_PSL"])
df["p2_odds_psw"]    = np.where(flip, df["d1_PSL"],            df["d1_PSW"])
df["p1_max_odds"]    = np.where(flip, df["d1_MaxL"],           df["d1_MaxW"])
df["p2_max_odds"]    = np.where(flip, df["d1_MaxW"],           df["d1_MaxL"])
df["p1_avg_odds"]    = np.where(flip, df["d1_AvgL"],           df["d1_AvgW"])
df["p2_avg_odds"]    = np.where(flip, df["d1_AvgW"],           df["d1_AvgL"])


# target: 1 = p1 won, 0 = p2 won
df["target"] = np.where(flip, 0, 1)


# onehot encode surface 
df = pd.get_dummies(df, columns=["surface"])
df = pd.get_dummies(df, columns=['tourney_name'])


# grab all of the newly generated columns 
surface_cols = [c for c in df.columns if c.startswith("surface_")]
tourney_cols = [c for c in df.columns if c.startswith("tourney_name_")]


# recent form of player
def get_recent_form(df, player_name, date, n=10):
    past = df[df['tourney_date'] < date]
    
    matches = past[
        (past['winner_name'] == player_name) |
        (past['loser_name'] == player_name)
    ].tail(n)
    
    if len(matches) == 0:
        return 0.5  # default to 50% if no history
    
    wins = len(matches[matches['winner_name'] == player_name])
    return wins / len(matches)

print("Computing recent form...")
p1_form, p2_form = [], []

for idx, row in df.iterrows():
    p1_name = row["winner_name"] if not flip[idx] else row["loser_name"]
    p2_name = row["loser_name"]  if not flip[idx] else row["winner_name"]
    
    p1_form.append(get_recent_form(df, p1_name, row['tourney_date']))
    p2_form.append(get_recent_form(df, p2_name, row['tourney_date']))

df["p1_form"] = p1_form
df["p2_form"] = p2_form


# features that will be used in model
feature_cols = [
    "p1_rank",
    "p2_rank",
    "p1_rank_points",
    "p2_rank_points",
    "p1_age",
    "p2_age",
    "p1_odds",
    "p2_odds",
    "surface_Hard",
    "surface_Clay",
    "surface_Grass",
    "p1_odds_psw",
    "p2_odds_psw",
    "p1_max_odds",
    "p2_max_odds",
    "p1_avg_odds",
    "p2_avg_odds",
    "p1_form",
    "p2_form"
] + surface_cols + tourney_cols

df = df.dropna(subset=feature_cols).reset_index(drop=True)


# Split data so that before November = train and after November = test 
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
df = df.sort_values('tourney_date').reset_index(drop=True)

X = df[feature_cols]
y = df["target"]

train_df = df[df['tourney_date'] < '2024-08-01']
test_df  = df[df['tourney_date'] >= '2024-08-01']

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]
y_train = train_df["target"]    
y_test  = test_df["target"]


# train 
model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.003,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)


# evaluate 
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds))


# feature importance 
importance = pd.Series(model.feature_importances_, index=feature_cols)
print("\nFeature importance:")
print(importance.sort_values(ascending=False))
