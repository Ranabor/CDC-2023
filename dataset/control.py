import sys

sys.path.append(".")
from authorization import get_auth_header
from requests import get
import json
import pandas as pd
from tqdm import tqdm


class SongsFromGenre:
    header = get_auth_header()

    def __init__(self) -> None:
        self.header = get_auth_header()

    def get_artist_from_song(self, song):
        url = f"https://api.spotify.com/v1/tracks/{song}"
        response = get(url, headers=self.header)
        json_response = response.json()
        artist_id = json_response["artists"][0]["uri"]
        artist_id = artist_id.replace("spotify:artist:", "")
        return artist_id

    def get_genre_from_artist(self, artist):
        url = f"https://api.spotify.com/v1/artists/{artist}"
        response = get(url, headers=self.header)
        json_response = response.json()
        genres = json_response["genres"]
        return genres

    # Commented out to not suffer
    # def find_top_3_genres(csv):
    #     songs_listing = pd.read_csv(csv)
    #     # new dictionary genres
    #     genres = {}
    #     for i in range(len(songs_listing)):
    #         song = songs_listing.iloc[i, 0]
    #         artist = get_artist_from_song(song)
    #         genre = get_genre_from_artist(artist)
    #         for i in range(len(genre)):
    #             if genre[i] not in genres:
    #                 genres[genre[i]] = 1
    #             else:
    #                 genres[genre[i]] += 1
    #     return genres

    # print(find_top_3_genres("dataset/final_songs.csv"))

    top_genres = {
        "colombian pop": 2,
        "dance pop": 131,
        "latin pop": 6,
        "pop": 106,
        "dancehall": 3,
        "pop rap": 73,
        "big room": 13,
        "edm": 45,
        "pop dance": 44,
        "art pop": 1,
        "atl hip hop": 32,
        "contemporary r&b": 6,
        "r&b": 27,
        "rap": 62,
        "south carolina hip hop": 4,
        "urban contemporary": 38,
        "eurodance": 81,
        "europop": 34,
        "german techno": 35,
        "melbourne bounce international": 2,
        "barbadian pop": 4,
        "latin hip hop": 7,
        "reggaeton": 5,
        "trap latino": 4,
        "urbano latino": 5,
        "dirty south rap": 39,
        "hip hop": 51,
        "old school atlanta hip hop": 12,
        "southern hip hop": 55,
        "gangster rap": 32,
        "st louis rap": 3,
        "east coast hip hop": 15,
        "queens hip hop": 9,
        "detroit hip hop": 4,
        "trap": 42,
        "canadian latin": 2,
        "canadian pop": 3,
        "miami hip hop": 20,
        "chicago rap": 3,
        "disco house": 8,
        "filter house": 9,
        "g funk": 6,
        "west coast rap": 7,
        "chicago bop": 3,
        "acid house": 2,
        "hip pop": 19,
        "finnish edm": 2,
        "neo soul": 1,
        "virginia hip hop": 1,
        "dutch house": 9,
        "electro house": 18,
        "house": 6,
        "progressive electro house": 18,
        "progressive house": 9,
        "romanian house": 2,
        "romanian pop": 3,
        "belgian pop": 3,
        "g-house": 1,
        "post-teen pop": 3,
        "asian american hip hop": 1,
        "nordic house": 1,
        "pop house": 3,
        "swedish electropop": 2,
        "swedish pop": 2,
        "mexican pop": 4,
        "deep dance pop": 1,
        "puerto rican pop": 1,
        "diva house": 24,
        "vocal house": 13,
        "belgian dance": 4,
        "hamburg electronic": 2,
        "happy hardcore": 3,
        "dream trance": 3,
        "trance": 10,
        "bouncy house": 8,
        "vocal trance": 3,
        "bubble trance": 10,
        "german trance": 4,
        "hardcore techno": 7,
        "uk dance": 3,
        "tribal house": 2,
        "alternative dance": 7,
        "big beat": 14,
        "electronica": 11,
        "scandipop": 1,
        "grime": 1,
        "instrumental grime": 1,
        "dance rock": 9,
        "indie rock": 1,
        "indietronica": 5,
        "neo-synthpop": 3,
        "new rave": 3,
        "hands up": 6,
        "italo dance": 8,
        "talent show": 2,
        "dutch pop": 2,
        "wrestling": 2,
        "freestyle": 5,
        "britpop": 1,
        "grebo": 1,
        "madchester": 2,
        "new wave pop": 4,
        "sophisti-pop": 1,
        "trip hop": 5,
        "miami bass": 4,
        "hip house": 22,
        "italo house": 1,
        "garage house": 1,
        "ambient house": 1,
        "album rock": 1,
        "classic rock": 2,
        "heartland rock": 1,
        "mellow gold": 1,
        "rock": 3,
        "singer-songwriter": 1,
        "soft rock": 1,
        "yacht rock": 1,
        "downtempo": 3,
        "gregorian dance": 2,
        "alternative rock": 1,
        "new romantic": 1,
        "new wave": 1,
        "permanent wave": 1,
        "post-punk": 1,
        "synthpop": 1,
        "uk post-punk": 1,
        "crunk": 13,
        "north carolina hip hop": 1,
        "atlanta bass": 1,
        "cali rap": 2,
        "golden age hip hop": 2,
        "hyphy": 2,
        "oakland hip hop": 1,
        "bounce": 4,
        "new orleans rap": 7,
        "futuristic swag": 6,
        "bronp hop": 1,
        "canadian hip hop": 1,
        "canadian old school hip hop": 2,
        "blues rock": 1,
        "mexican classic rock": 1,
        "swedish idol pop": 1,
        "bubblegum dance": 5,
        "swedish country": 2,
        "german pop": 4,
        "cologne hip hop": 1,
        "oldschool deutschrap": 1,
        "boy band": 5,
        "girl group": 2,
        "german alternative rock": 1,
        "german pop rock": 1,
        "tropical": 1,
        "moldovan pop": 1,
        "italian adult pop": 1,
        "reggae": 1,
        "reggae fusion": 4,
        "alternative metal": 2,
        "funk metal": 2,
        "nu metal": 1,
        "post-grunge": 2,
        "rap rock": 2,
        "old school hip hop": 1,
        "pop rock": 1,
        "bahamian pop": 1,
        "comic": 1,
        "funk rock": 1,
        "disco": 1,
        "hi-nrg": 1,
        "german dance": 4,
        "brostep": 2,
        "complextro": 15,
        "electro": 3,
        "metropopolis": 2,
        "uk pop": 1,
        "breakbeat": 7,
        "rave": 7,
        "dutch trance": 3,
        "dance-punk": 2,
        "classic house": 3,
        "electropop": 1,
        "nantes indie": 1,
        "french shoegaze": 1,
        "french synthpop": 1,
        "canadian electronic": 1,
        "electroclash": 1,
        "minimal techno": 1,
        "full on": 1,
        "nitzhonot": 1,
        "psychedelic trance": 2,
        "italian trance": 1,
        "nu skool breaks": 1,
        "chicago house": 2,
        "classic progressive house": 1,
        "dutch edm": 2,
        "slap house": 1,
        "afropop": 1,
        "griot": 1,
        "guinean pop": 1,
        "mande pop": 1,
        "west african jazz": 1,
        "techhouse": 1,
        "uk house": 1,
        "hip hop tuga": 1,
        "portuguese pop": 1,
        "electronic rock": 1,
        "hard trance": 1,
        "deep house": 1,
        "chutney": 1,
        "soca": 1,
        "philly rap": 1,
        "australian dance": 1,
        "australian pop": 1,
        "electro latino": 1,
        "spanish pop": 1,
        "bass music": 1,
        "nz pop": 1,
        "desi pop": 1,
        "kids dance party": 1,
        "kansas city hip hop": 1,
    }

    top_5_genres = sorted(top_genres, key=top_genres.get, reverse=True)[:3]

    def get_top_artists(self, genre):
        url = f"https://api.spotify.com/v1/search?q=genre:%22{genre}%22&type=artist&limit=5"
        response = get(url, headers=self.header)
        json_response = response.json()
        artists = json_response["artists"]["items"]
        df1 = pd.json_normalize(artists)
        df2 = df1[["uri", "name"]]
        df2["uri"] = df2["uri"].str.replace("spotify:artist:", "")
        return df2

    # for a given artist, take their top albums released between 2000 and 2010
    def get_albums_from_artist(self, artist):
        url = f"https://api.spotify.com/v1/artists/{artist}/albums?market=US&limit=50"
        response = get(url, headers=self.header)
        json_response = response.json()
        albums = json_response["items"]
        df1 = pd.json_normalize(albums)
        df1 = df1[df1["release_date"] >= "2000-01"]
        df1 = df1[df1["release_date"] <= "2010-12"]
        df2 = df1[["uri", "name"]]
        df2["uri"] = df2["uri"].str.replace("spotify:album:", "")
        return df2

    # find a song's popularity
    def get_song_popularity(self, song):
        url = f"https://api.spotify.com/v1/tracks/{song}"
        response = get(url, headers=self.header)
        json_response = response.json()
        popularity = json_response["popularity"]
        # numeric value
        return int(popularity)

    # from an album, get all the songs and put in a dataframe
    def get_songs_from_album(self, album, csv):
        url = f"https://api.spotify.com/v1/albums/{album}/tracks?market=US&limit=50"
        response = get(url, headers=self.header)
        json_response = response.json()
        songs = json_response["items"]
        df1 = pd.json_normalize(songs)
        df2 = df1["uri"].str.replace("spotify:track:", "")
        df3 = pd.read_csv(csv)
        for i in range(len(df3)):
            song = df3.iloc[i, 0]
            if song in df2:
                df2 = df2.drop(df2[df2 == song].index)
        return df2


GenreHelper = SongsFromGenre()


def get_songs_from_genre(genre):
    artists = GenreHelper.get_top_artists(genre)
    songs = pd.DataFrame()
    for i in range(len(artists)):
        artist = artists.iloc[i, 0]
        albums = GenreHelper.get_albums_from_artist(artist)
        for j in range(len(albums)):
            album = albums.iloc[j, 0]
            songs_from_album = GenreHelper.get_songs_from_album(
                album, "dataset/final_songs.csv"
            )
            songs = pd.concat([songs, songs_from_album])
    songs = songs.drop_duplicates()
    return songs


# for each genre in top 5 genres, get the songs. add to a dataframe. then output to csv
songs_from_genres = pd.DataFrame()
bad_genres = ["classical", "country", "jazz"]
for i in tqdm(range(len(GenreHelper.top_5_genres))):
    genre = GenreHelper.top_5_genres[i]
    songs = get_songs_from_genre(genre)
    songs_from_genres = pd.concat([songs_from_genres, songs])
songs_from_genres = songs_from_genres.drop_duplicates()
# songs_from_genres.to_csv("dataset/non_bangers_good_genres.csv", index=False)
