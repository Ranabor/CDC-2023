# imports
import sys

sys.path.append(".")
from authorization import get_auth_header
from requests import get
import json
import pandas as pd


# write a function to get the top 20 artists in the pop genre using the spotify api example
def get_top_artists():
    # get the token
    # get the header
    header = get_auth_header()
    # get the url
    url = "https://api.spotify.com/v1/search?q=genre:%22pop%22&type=artist&limit=20"
    # get the response
    response = get(url, headers=header)
    # convert to json
    json_response = response.json()
    # get the artists
    artists = json_response["artists"]["items"]
    # return the artists
    return artists


def get_song_from_playlist(playlist):
    header = get_auth_header()
    url = (
        "https://api.spotify.com/v1/playlists/"
        + playlist
        + "/tracks?market=US&limit=100"
    )
    response = get(url, headers=header)
    json_response = response.json()
    songs = json_response["items"]
    df1 = pd.json_normalize(songs)  # created dataframe with all data/columns
    df2 = df1[
        ["track.uri", "track.name"]
    ]  # creates anotehr df with only the track names
    df2["track.uri"] = df2["track.uri"].str.replace("spotify:track:", "")
    return df2  # return df with track names


playlist_ids = [
    "37i9dQZF1DX7e8TjkFNKWH",
    "4vKtucORuHzl3bhnUWMMbq",
    "54nOF0dt7yuHMBK1cLjFCk",
    "1q4yqfWq8DQ9k8xvpwjSGb",
    "66bd2w90q5RXlfUnTIF1W3",
    "2Hy1V8q07RL4ygQtatadve",
    "33P9WHZFJArsEEb0q8zwJD",
    "3EoStzWMnc93ktTEZYsSD4",
    "5KlUhhSR7sZOdl8Hxy3Guz",
    "6ebBexShOwJOfK9UIdzNIm",
]
songs_list = pd.DataFrame()
for i in range(len(playlist_ids)):
    playlist_songs = get_song_from_playlist(playlist_ids[i])
    songs_list = pd.concat([songs_list, playlist_songs])
songs_list = songs_list.drop_duplicates()
songs_list


def compare_songs(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df3 = df1[df1["track.name"].isin(df2["track.name"])]
    return df3


final_songs = compare_songs(
    "./dataset/banger_songs_list_test.csv", "./dataset/banger_songs_list.csv"
)
# final_songs.to_csv("final_songs.csv", index=False)
