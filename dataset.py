from authorization import get_auth_header
from requests import get
import json

def get_2000s_club_playlists():
    url = 'https://api.spotify.com/v1/search?q=2000s+club+hits&type=playlist&market=ES&limit=10'
    result = get(url, headers=get_auth_header())
    json_result = json.loads(result.text)
    print(json_result)
    
    # json_result = json.dumps()
    # with open("sample.json", "w") as outfile:
    #     outfile.write(json_object)
    
    
get_2000s_club_playlists()