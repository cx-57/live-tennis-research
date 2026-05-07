import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("merged.csv")


# drop rows with missing values from just winner and loser rank right now
df = df.dropna(subset = ["winner_rank", "loser_rank"]).reset_index(drop=True)


# randomly flip rows so model doesn't just learn "player 1 always wins"
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


# target: 1 = p1 won, 0 = p2 won
df["target"] = np.where(flip, 0, 1)


# onehot encode surface 
df = pd.get_dummies(df, columns=["surface"])

print([c for c in df.columns if "surface" in c.lower()])


# features
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
]

df = df.dropna(subset=feature_cols).reset_index(drop=True)

X = df[feature_cols]
y = df["target"]

# split data
split = int(len(df) * 0.8)  # no train test split, because data must be chronological
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# train
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)


# evaluate 
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(classification_report(y_test, preds))


# feature importance 
importance = pd.Series(model.feature_importances_, index=feature_cols)
print("\nFeature importance:")
print(importance.sort_values(ascending=False))