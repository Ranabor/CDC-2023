#imports
from authorization import get_auth_header, get_token
from requests import get
import json
import pandas as pd

#write a function to get the top 20 artists in the pop genre using the spotify api example
def get_top_artists():
    #get the token
    token = get_token()
    #get the header
    header = get_auth_header(token)
    #get the url
    url = "https://api.spotify.com/v1/search?q=genre:%22pop%22&type=artist&limit=20"
    #get the response
    response = get(url, headers=header)
    #convert to json
    json_response = response.json()
    #get the artists
    artists = json_response['artists']['items']
    #return the artists
    return artists

def get_song_from_playlist(playlist):
    token = get_token()
    header = get_auth_header(token)
    url = "https://api.spotify.com/v1/playlists/" + playlist + "/tracks?market=US&limit=100&offset=100"
    response = get(url, headers=header)
    json_response = response.json()
    songs = json_response['items']
    df1 = pd.json_normalize(songs) #created dataframe with all data/columns
    df2 = df1[['track.uri', "track.name"]] #creates anotehr df with only the track names
    df2['track.uri'] = df2['track.uri'].str.replace("spotify:track:", "")
    return df2 #pd.DataFrame(df2) #return df with track names


playlist_ids = ["37i9dQZF1DX7e8TjkFNKWH", "4vKtucORuHzl3bhnUWMMbq", "54nOF0dt7yuHMBK1cLjFCk",
                  "1q4yqfWq8DQ9k8xvpwjSGb", "66bd2w90q5RXlfUnTIF1W3", "2Hy1V8q07RL4ygQtatadve", 
                  "33P9WHZFJArsEEb0q8zwJD", "3EoStzWMnc93ktTEZYsSD4", "5KlUhhSR7sZOdl8Hxy3Guz", 
                  "6ebBexShOwJOfK9UIdzNIm"]
songs_list = pd.DataFrame()
for i in range (len(playlist_ids)):
    playlist_songs = get_song_from_playlist(playlist_ids[i])
    songs_list = pd.concat([songs_list, playlist_songs])
songs_list = songs_list.drop_duplicates()
songs_list
songs_list.to_csv('banger_songs_list.csv', index=False)
