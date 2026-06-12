# Live win probability

This folder models the probability that a player wins a match while the match is still in
progress, updated after every point. The pre-match prediction problem, where the goal is to call
the winner before play starts, is handled elsewhere in this repository. Here the score is
unfolding and the question is who is winning right now, given everything that has happened so far.

The data is the Grand Slam point-by-point record published by Jeff Sackmann, covering the four
majors from 2011 to 2024. Each point gives the score after it was played and who served it, which
is exactly the information available live. The label on every row is whether player 1 went on to
win the match, so a model learns to map an in-progress score to a probability.

## Data and preparation

Three public datasets are needed. The Grand Slam point-by-point files provide the live scores.
The ATP and WTA match results provide the tour history used to rate players before a match. Clone
them into the `data` folder so that `data/slam`, `data/atp` and `data/wta` each point at the
corresponding repository:

```
git clone https://github.com/JeffSackmann/tennis_slam_pointbypoint data/slam
git clone https://github.com/JeffSackmann/tennis_atp data/atp
git clone https://github.com/JeffSackmann/tennis_wta data/wta
```

`prepare_data.py` reads the point-by-point files and writes one row per point with the score
state, the derived match winner and best-of length, and each player's running serve-won rate
computed from earlier points only. `elo.py` builds a pre-match rating for every player from the
full tour history and attaches it to each slam match. Both scripts write to the `artifacts`
folder, which the models read.

The evaluation uses a time split so that no match is seen before it happens. Matches up to 2021
are used for training, the 2022 season for calibrating the handful of free parameters, and the
2023 and 2024 seasons are held out for the final numbers reported below.

## The models

The starting point is a structural baseline rather than a fitted one. `baseline_markov.py` treats
points as independent and the two players as equally strong, so the live win probability is a pure
function of the score. With a single serve-win probability it computes, through a nested
point to game to set to match recursion, the exact chance that player 1 wins from any score. For
example, two sets to love ahead in a best-of-five comes out at 0.875, which is one minus a half
cubed, since one of the next three sets is enough. The only free number is the serve-win
probability, chosen on the validation season. This model says that the score is the state and
nothing about the path to it matters.

`baseline_gbm.py` is the flexible counterpart. It trains gradient-boosted trees on the same score
features and learns the map from score to win probability directly from data, without assuming the
recursion. It is useful as a check on how much a general purpose learner can extract from the score
alone.

`asymmetric_markov.py` keeps the recursion but drops the assumption that the players are equal.
Each player is given a different serve-win probability, set by the pre-match Elo gap as a base
value plus or minus a slope times the rating difference. A stronger player gets a higher serve
probability and the opponent a lower one, so a match between a top seed and a qualifier starts well
away from fifty-fifty instead of at the equal-players value. The base and slope are calibrated on
the validation season. This is the single change that gives the largest gain, because it fixes the
otherwise blank starting point of every match.

`serve_shrink_model.py` is the full model. It recognises that as a match goes on, the players
reveal how well they are actually serving today, which the pre-match rating cannot know. Each
player's serve-win probability becomes a blend of the Elo-implied prior and the serve-won rate
observed so far in the match. The blend is a Bayesian shrink controlled by a pseudo-count: while
few serve points have been seen the prior dominates, and as serve points accumulate the observed
rate takes over. The recursion then turns the two serve probabilities and the current score into a
win probability as before. The calibrated pseudo-count is large, which says the observed serve
rate is noisy and should be trusted only once a good deal of it has been seen.

## Results

Held-out test seasons 2023 and 2024, roughly 180,000 points across 969 matches.

| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| Symmetric Markov anchor | 0.5294 | 0.1758 | 0.7312 |
| Gradient boosting on score | 0.5159 | 0.1735 | 0.7345 |
| Asymmetric Markov with Elo prior | 0.4663 | 0.1516 | 0.7755 |
| Serve-shrink model | 0.4600 | 0.1497 | 0.7766 |

The score alone, read through the recursion, already reaches 73 percent accuracy, and a gradient
boosting model given the same score features does only a little better, which says the score is
most of the signal. The large step comes from the pre-match rating, which lifts accuracy past 77
percent by giving each match a sensible starting point. Adding the in-match serving on top of that
gives a further, smaller improvement. The serve-shrink model uses three interpretable numbers, a
base serve probability, a slope from rating to serve advantage, and the shrink pseudo-count, and it
matches a far larger gradient boosting model that is given every feature, while staying readable
and giving a probability for any score that can be inspected by hand.

## Running it

From this folder, with the three datasets cloned into `data` and the dependencies installed
(`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`):

```
python prepare_data.py
python elo.py
python baseline_markov.py
python baseline_gbm.py
python asymmetric_markov.py
python serve_shrink_model.py
```

The first two scripts build the artifacts and only need to be run once. Each model script
calibrates its parameters on the 2022 season and prints its metrics on the held-out seasons.
