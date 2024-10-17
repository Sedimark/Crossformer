"""
    Description: Data Interface
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class Data_Weather(Dataset):
    """
        For the SEDIMARK weather data use only
    """
    def __init__(self, df,
                 size=[24,24]) -> None:
        super().__init__()

        self.df = df
        
        # convert into np
        if self.df.columns[0] != 'UnixTime':
            self.df = self.df.values[:,1:] # remove the index column
        else:
            self.df = self.df.to_numpy()

        # input and output lengths
        self.in_len = size[0]
        self.out_len = size[1]

        chunk_size = self.in_len + self.out_len
        chunk_num = len(self.df) // chunk_size

        self.chunks = {}
        self._prep_chunk(chunk_size, chunk_num)

    def _prep_chunk(self, chunk_size, chunk_num):
        for i in range(chunk_num):
            chunk_data = self.df[i*chunk_size:(i+1)*chunk_size,:]
            self.chunks[i] = {
                'feat_data': chunk_data[:self.in_len,1:],
                'target_data': chunk_data[-self.out_len:,1:],
                'annotation': {
                    'feat_time': chunk_data[:self.in_len,0],
                    'target_time': chunk_data[-self.out_len:,0],
                }
            }

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]
    
def custom_collate_fn(batch):
        feat_batch = []
        target_batch = []
        annotation_batch = []

        for item in batch:
            feat_batch.append(torch.tensor(item['feat_data'], dtype=torch.float32))
            target_batch.append(torch.tensor(item['target_data'], dtype=torch.float32))
            annotation_batch.append(item['annotation'])

        feat_batch = torch.stack(feat_batch)
        target_batch = torch.stack(target_batch)

        return feat_batch, target_batch, annotation_batch

class DataInterface(pl.LightningDataModule):
    def __init__(self, df, size=[24,24], split=[0.7,0.2,0.1],
                 batch_size=1, **kwargs) -> None:
        super().__init__()
        self.df = df
        self.split = split
        self.size = size
        self.batch_size = batch_size

    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        dataset = Data_Weather(self.df)
        self.train, self.val, self.test = random_split(dataset, self.split)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=custom_collate_fn, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=custom_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=custom_collate_fn)
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('/home/paul/CODE/fl_crossformer/broker.csv')
    data = DataInterface(df)
    data.setup()
    train_loader = data.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(batch)
        if i == 0:
            break
