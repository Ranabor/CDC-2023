import pandas as pd
from scipy import stats

bangers = pd.read_csv('banger_features.csv').drop(['track.name'], axis=1)
nonbangers = pd.read_csv('nonbanger_features.csv')

bangers['is_banger'] = 'yes'
nonbangers['is_banger'] = 'no'

final_dataset = pd.concat([bangers, nonbangers], axis=0, ignore_index=True).drop(['Unnamed: 0'], axis=1)

columns = list(final_dataset.columns[1:-1])

for column in columns:
    final_dataset[column] = stats.zscore(final_dataset[column])

final_dataset.to_csv('audio_features.csv', index=False)

