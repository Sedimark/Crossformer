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
from crossformer.utils.tools import scaler

class Genera_Data(Dataset):
    """
        *** For the general SEDIMAKR data use ***
        SEDIMARK Data is structured in annotations (etc. data type, collumn name) and values. 
        As this class is used for core wheels (or we also called core functions), we do only accept values as input.
        Therefore, an external tool for linking annotations and values will be provided in future.
    """
    def __init__(self, df, 
                 size=[24,24],
                 **kwargs):
        """
            Args:
                df: pandas dataframe (values only)
                size: list of two integers, [input_length, output_length]
        """
        super().__init__()
        
        self.values = df.to_numpy()
        self.in_len, self.out_len = size

        chunk_size = self.in_len + self.out_len
        chunk_num = len(self.values) // chunk_size

        self.chunks = {}
        self._prep_chunk(chunk_size, chunk_num)

    def _prep_chunk(self, chunk_size, chunk_num):
        for i in range(chunk_num):
            chunk_data = self.values[i*chunk_size:(i+1)*chunk_size,:]
            scale, base = scaler(chunk_data[:self.in_len])
            self.chunks[i] = {
                'feat_data': base,
                'scale': scale,
                'target_data': chunk_data[-self.out_len:],
            }

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx]['feat_data'], dtype=torch.float32), torch.tensor(self.chunks[idx]['scale'], dtype=torch.float32), torch.tensor(self.chunks[idx]['target_data'], dtype=torch.float32)

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
        dataset = Genera_Data(self.df, size=self.size)
        self.train, self.val, self.test = random_split(dataset, self.split)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,)
    
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('././all_weather_values.csv')
    data = DataInterface(df, size=[2,2],batch_size=1)
    data.setup()
    train_loader = data.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(batch)
        if i == 0:
            break
