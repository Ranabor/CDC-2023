import torch
from torch.utils.data import Dataset
import pandas as pd

class SongDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.df = pd.read_csv('audio_features.csv')
        sig_features = [1, 2, 3, 4, 6, 7, 9, 10, 13]
        self.df = self.df.iloc[:, sig_features]
        self.columns = list(self.df.columns)[:-1]
    
    def __getitem__(self, idx):
        features = []
        for column in self.columns:
            features.append(self.df[column].iloc[idx])
        item = {
            'features': torch.Tensor(features),
            'labels': self.df['club_compatibility'].iloc[idx]
        }
        return item
        
        
    def __len__(self):
        return len(self.df)