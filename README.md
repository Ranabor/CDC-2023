https://devpost.com/software/club-compatibility


- api: https://developer.spotify.com/documentation/web-api
- https://open.spotify.com/playlist/7fmD71MFQJNVwWFAho5aPV?si=c1fcd3cc2b3f4f25
- [ ] Make Dataset
  - [ ] find features we want to use
- [ ] make data viz
  - [ ] find chart ideas
- [ ] make model to predit "bangability"
  - [ ] use LLM
    - [ ] pass in the features and embeddings
    - [ ] "is it bangable?" look at probability of yes
  - [ ] look at cover art

Control: Top 20 artists of club banger genre songs
Experimental: Club bangers

Playlist IDs
- 37i9dQZF1EIezLFyG0SSkJ
- 4vKtucORuHzl3bhnUWMMbq
- 54nOF0dt7yuHMBK1cLjFCk
- 1q4yqfWq8DQ9k8xvpwjSGb
- 66bd2w90q5RXlfUnTIF1W3
- 2Hy1V8q07RL4ygQtatadve
- 33P9WHZFJArsEEb0q8zwJD
- 3EoStzWMnc93ktTEZYsSD4
- 5KlUhhSR7sZOdl8Hxy3Guz
- 6ebBexShOwJOfK9UIdzNIm

Non-bangers dataset

Dataset 1: In-Genre
- Retrieve top 20 artists for the most common banger genres
- Get all the songs for the picked artists
- Filter out songs that are part of the bangers dataset
- Take out non-bangers with more than 60% popularity
- Get the audio features

Dataset 2: Out-of-Genre
- Out-of-Genre: Country, classical, reggae, jazz, blues
- Find playlists (1-2) for the genres
- Get the audio features

API Feature Analysis
acousticness,
danceability,
duration_ms,
energy,
instrumentalness,
liveness,
loudness,
speechiness,
tempo,
valence,
key,
time_signature




