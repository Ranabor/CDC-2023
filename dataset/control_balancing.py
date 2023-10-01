# imports
import sys

sys.path.append(".")
from authorization import get_auth_header
from requests import get
import json
import pandas as pd

good_genres = pd.read_csv("dataset/non_bangers_good_genres.csv")
bad_genres = pd.read_csv("dataset/non_bangers_bad_genres.csv")

# randomly remove 1/4 of good genre songs
good_genres = good_genres.sample(frac=0.75)

# concatenate good and bad
non_bangers = pd.concat([good_genres, bad_genres])

# csv output
non_bangers.to_csv("dataset/non_bangers.csv", index=False)
