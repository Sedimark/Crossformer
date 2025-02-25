"""Data Interface.

    Data Interface provides dataset and datamodule definition.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from lightning.pytorch import LightningDataModule
from crossformer.utils.tools import scaler


class General_Data(Dataset):
    """
    *** For the general SEDIMARK data use ***
    SEDIMARK Data is structured in annotations (etc. data type, collumn name)
    and values. As this class is used for core wheels (or we also called core
    functions), we do only accept values as input. Therefore, an external tool
    for linking annotations and values will be provided in future.
    """

    def __init__(self, df, size=[24, 24], **kwargs):
        """_initialize the general dataset class

        Args:
            df (pd.DataFrame): DataFrame (data).
            size (list): Chunk size [input length, output length]. Defaults to
                         [24, 24].
        """
        super().__init__()

        self.values = df.to_numpy()
        self.in_len, self.out_len = size

        chunk_size = self.in_len + self.out_len
        chunk_num = len(self.values) // chunk_size

        self.chunks = {}
        self._prep_chunk(chunk_size, chunk_num)

    def _prep_chunk(self, chunk_size, chunk_num, normlization=False):
        """Split chunks.

        Args:
            chunk_size (int): The chunk size.
            chunk_num (int): The chunk num.
            normlization (bool, optional): Control normalization.
                                           Defaults to False.
        """
        for i in range(chunk_num):
            chunk_data = self.values[i * chunk_size : (i + 1) * chunk_size, :]  # noqa: E203, E501
            if normlization:
                scale, base = scaler(chunk_data[: self.in_len])  # noqa: E203
            else:
                scale, base = (
                    0,
                    chunk_data[: self.in_len],
                )  # chunk_data will be processed directly
            self.chunks[i] = {
                'feat_data': base,
                'scale': scale,
                'target_data': chunk_data[-self.out_len :],  # noqa: E203
            }

    def __len__(self):
        """Return the length of dataset.

        Returns:
            (int): The length of dataset
        """
        return len(self.chunks)

    def __getitem__(self, idx):
        """Get item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            (torch.tensor): Item's content. (feat_data, scale, ground_truth)
        """
        return (
            torch.tensor(self.chunks[idx]['feat_data'], dtype=torch.float32),
            torch.tensor(self.chunks[idx]['scale'], dtype=torch.float32),
            torch.tensor(self.chunks[idx]['target_data'], dtype=torch.float32),
        )


class DataInterface(LightningDataModule):
    """Data Interface.

    It supports pytorch lightning trainer to call the data module.
    """

    def __init__(
        self,
        df,
        in_len=24,
        out_len=24,
        split=[0.7, 0.2, 0.1],
        batch_size=1,
        num_workers=31,
        **kwargs,
    ) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): DataFrame (data).
            in_len (int): The length of the input sequence.
            out_len (int): The length of the output sequence.
            split (list, optional): Splits of train, val and test set.
                                    Defaults to [0.7, 0.2, 0.1].
            batch_size (int, optional): Batch size. Defaults to 1.
        """
        super().__init__()
        self.df = df
        self.split = split
        self.size = [in_len, out_len]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            dataset = General_Data(self.df, size=self.size)
            # based on time
            train_num, test_num = int(dataset.__len__() * self.split[0]), int(
                dataset.__len__() * self.split[2]
            )
            val_num = dataset.__len__() - train_num - test_num
            train_index, val_index, test_index = (
                list(range(0, train_num)),
                list(range(train_num, train_num + val_num)),
                list(range(train_num + val_num, dataset.__len__())),
            )
            self.train, self.val, self.test = (
                Subset(dataset=dataset, indices=train_index),
                Subset(dataset=dataset, indices=val_index),
                Subset(dataset=dataset, indices=test_index),
            )

        if stage == 'predict':
            self.predict = General_Data(self.df, size=self.size)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
        )


# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv('././all_weather_values.csv')
#     data = DataInterface(df, size=[2, 2], batch_size=1)
#     data.setup()
#     train_loader = data.train_dataloader()
#     for i, batch in enumerate(train_loader):
#         print(batch)
#         if i == 0:
#             break
