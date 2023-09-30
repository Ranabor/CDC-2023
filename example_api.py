from authorization import get_auth_header, get_token
from requests import get
import json

token = get_token()
headers = get_auth_header(token)


def search_for_artist(token, name):
    url = "https://api.spotify.com/v1/search"
    query = f"?q={name}&type=artist&limit=1"
    result = get(url + query, headers=get_auth_header(token))
    json_result = json.loads(result.content)["artists"]["items"]
    if len(json_result) == 0:
        print("none found")
        return None
    return json_result[0]


result = search_for_artist(token, "Drake")
print(result)
