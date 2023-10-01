#imports
import sys
sys.path.append(".")
from authorization import get_auth_header, get_token
from requests import get
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


csv_file = "dataset/final_songs.csv"



def get_audio_features(csv_file):
    feature_columns = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'key', 'time_signature']
    feature_df = pd.DataFrame(columns=feature_columns)
    

    df = pd.read_csv(csv_file) #df of csv file; csv file should have two columns: uri and name
    id_list = df['track.uri'].tolist()
    header = get_auth_header()

    for i in range(0, len(df), 100): #can only get audio features 100 at a time, comma seperated
        LOWER = i
        UPPER = i+99 if i + 99 < len(df) else len(df)
        temp_list = id_list[LOWER:UPPER+1] #subset of elements from id_list to fit in 100 req
        print(len(temp_list))
        #converting temp_list to a string to be read by url link, farmated as 'id, id, ' and so on...
        temp_str = ""
        for  id in temp_list:
            temp_str += f"{id},"
        temp_str = temp_str[:-1] #deletes last extra comma
        
        
        url = f"https://api.spotify.com/v1/audio-features?ids={temp_str}"
        response = get(url, headers=header)
        response = json.loads(response.text)
        for song in response["audio_features"]:
            feature_list = [song[feature] for feature in feature_columns]
            feature_df.loc[len(feature_df)] = feature_list
        
    final_df = pd.concat([df, feature_df], axis=1)
    #songs = json_response['items']
    return final_df

final_df = get_audio_features(csv_file)

final_df.to_csv('banger_features.csv', index=False)
"""""""""
# Assuming df is your DataFrame with 'Column1' and 'Column2'
plt.figure(figsize=(10, 6))
sns.scatterplot(x=final_df.index.tolist(), y='valence', data=final_df)
# Set plot title and axis labels
plt.title('Scatter Plot of Column1 vs Column2')
plt.xlabel('Column1')
plt.ylabel('Column2')

# Display the plot
plt.show()
"""

""""
# Plot the histogram
final_df = csv_reader
final_df['valence'].hist(bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Songs')
plt.ylabel('Valence')
plt.title('Histogram of Values')
plt.show()
"""