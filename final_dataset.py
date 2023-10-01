import pandas as pd

bangers = pd.read_csv('banger_features.csv').drop(['track.name'], axis=1)
nonbangers = pd.read_csv('nonbanger_features.csv')

bangers['is_banger'] = 'yes'
nonbangers['is_banger'] = 'no'

final_dataset = pd.concat([bangers, nonbangers], axis=0, ignore_index=True).drop(['Unnamed: 0'], axis=1)
final_dataset.to_csv('audio_features.csv', index=False)