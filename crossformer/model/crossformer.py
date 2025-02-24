"""Crossformer.

    CrossFormer model definition and module implementaiton.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

from math import ceil
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from einops import repeat
from crossformer.model.layers.encoder import Encoder
from crossformer.model.layers.decoder import Decoder
from crossformer.model.layers.embedding import ValueEmebedding
from crossformer.utils.metrics import metric, hybrid_loss

class Crossformer(nn.Module):
    """Crossformer class.

    The implementation based on pytorch.
    """

    def __init__(
        self,
        data_dim,
        in_len,
        out_len,
        seg_len,
        window_size=4,
        factor=10,
        model_dim=512,
        feedforward_dim=1024,
        heads_num=8,
        blocks_num=3,
        dropout=0.0,
        baseline=False,
        **kwargs,
    ):
        """Initialize the Crossformer class.

        Args:
            data_dim (int): The dimension of the input data.
            in_len (int): The length of the input sequence.
            out_len (int): The length of the output sequence.
            seg_len (int): The length of the segment.
            window_size (int, optional): The size of the window. Defaults to 4.
            factor (int, optional): The factor of the sparse attention.
                                    Defaults to 10.
            model_dim (int, optional): The dimension of the model.
                                       Defaults to 512.
            feedforward_dim (int, optional): The dimension of the
                                             feedforward network.
                                             Defaults to 1024.
            heads_num (int, optional): The number of heads. Defaults to 8.
            blocks_num (int, optional): The number of blocks. Defaults to 3.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            baseline (bool, optional): Whether to use the baseline.
                                       Defaults to False.
        """
        super(Crossformer, self).__init__()

        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = window_size

        self.baseline = baseline

        # Segment Number alculation
        self.in_seg_num = ceil(1.0 * in_len / seg_len)
        self.out_seg_num = ceil(1.0 * out_len / seg_len)

        # Encode Embedding & Encoder
        self.enc_embedding = ValueEmebedding(
            seg_len=self.seg_len, model_dim=model_dim
        )
        self.enc_pos = nn.Parameter(
            torch.randn(1, data_dim, (self.in_seg_num), model_dim)
        )
        self.norm = nn.LayerNorm(model_dim)
        self.encoder = Encoder(
            blocks_num=blocks_num,
            model_dim=model_dim,
            window_size=window_size,
            depth=1,
            seg_num=self.in_seg_num,
            factor=factor,
            heads_num=heads_num,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        # Decode Embedding & Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, (self.out_seg_num), model_dim)
        )
        self.decoder = Decoder(
            seg_len=self.seg_len,
            model_dim=model_dim,
            heads_num=heads_num,
            depth=1,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            out_segment_num=self.out_seg_num,
            factor=factor,
        )

    def forward(self, x_seq):
        """Forward pass data to the Crossformer model.

        Args:
            x_seq (torch.Tensor): Input sequence tensor of shape
                      (batch_size, timeseries_length, timeseries_dim).

        Returns:
            torch.Tensor: Output sequence tensor of shape
                  (batch_size, out_len, timeseries_dim).
        """

        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if self.in_seg_num * self.seg_len != self.in_len:
            x_seq = torch.cat(
                (
                    x_seq[:, :1, :].expand(
                        -1, (self.seg_len * self.in_seg_num - self.in_len), -1
                    ),
                    x_seq,
                ),
                dim=1,
            )

        x_seq = self.enc_embedding(x_seq)
        x_seq += self.enc_pos
        x_seq = self.norm(x_seq)

        enc_out = self.encoder(x_seq)
        dec_in = repeat(
            self.dec_pos_embedding,
            'b ts_d l d -> (repeat b) ts_d l d',
            repeat=batch_size,
        )
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, : self.out_len, :]


class CrossFormer(LightningModule):
    """CrossFormer

    The implementation based on Lightning with the engineering code.
    """

    def __init__(self, cfg=None, learning_rate=1e-4, batch=32, **kwargs):
        """_initialize the Lightning module (CrossFormer)

        Args:
            cfg (dict): Configuration. Defaults to None.
            learning_rate (float, optional): Learning rate. Defaults to 1e-4.
            batch (int, optional): Batch number. Defaults to 32.
        """
        super(CrossFormer, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        # Create the model
        self.model = Crossformer(**cfg)

        # Training Parameters
        # self.loss = nn.MSELoss()
        self.loss = hybrid_loss
        self.learning_rate = learning_rate
        self.batch = batch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step.

        Forward pass and calculate loss.

        Args:
            batch (torch.tensor): Batch data.
            batch_idx (int): Batch index.

        Returns:
            (dict): logging the loss.
        """
        (x, scale, y) = batch
        if scale._is_zerotensor():
            y_hat = self(x) * scale.unsqueeze(1)
        else:
            y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Forward pass and calculate metrics.

        Args:
            batch (torch.Tensor): Batch data.
            batch_idx (int): Batch index.

        Returns:
            dict: Validation metrics.
        """
        (x, scale, y) = batch
        if scale._is_zerotensor():
            y_hat = self(x) * scale.unsqueeze(1)
        else:
            y_hat = self(x)
        metrics = metric(y_hat, y)
        metrics = {f'val_{key}': value for key, value in metrics.items()}
        self.log_dict(
            metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return metrics

    def test_step(self, batch, batch_idx):
        """Test step.

        Forward pass and calculate metrics.

        Args:
            batch (torch.Tensor): Batch data.
            batch_idx (int): Batch index.

        Returns:
            dict: Test metrics.
        """
        (x, scale, y) = batch
        if scale._is_zerotensor():
            y_hat = self(x) * scale.unsqueeze(1)
        else:
            y_hat = self(x)
        metrics = metric(y_hat, y)
        metrics = {f'test_{key}': value for key, value in metrics.items()}
        self.log_dict(
            metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return metrics

    def predict_step(self, batch, *args, **kwargs):
        """Predict step.

        Forward pass and return predictions.

        Args:
            batch (torch.Tensor): Batch data.

        Returns:
            torch.Tensor: Predicted values.
        """
        (x, scale, y) = batch
        y_hat = self(x) * scale.unsqueeze(1)
        return y_hat

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.1 ** (epoch // 20)
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_MAE',
                'interval': 'epoch',
                'frequency': 1,
            },
        }