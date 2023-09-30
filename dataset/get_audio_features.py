#imports
from ..authorization import get_auth_header, get_token
from requests import get
import json
import pandas as pd


def get_audio_features(csv_file):
    df = pd.read_csv(csv_file)
    header = get_auth_header()
    for i in range(0, len(df), 100):
        LOWER = i
        UPPER = i+99 if i + 99 < len(df) else len(df) - 1
        temp_df= df.loc[LOWER:UPPER, : ]
        print(temp_df.tostring())
        url = "https://api.spotify.com/v1/audio-features/{temp_df}"
        response = get(url, headers=header)
    json_response = response.json()
    songs = json_response['items']
