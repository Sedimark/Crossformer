"""
    Description: CrossFormer model
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat

from crossformer.model.layers.attention import TwoStageAttentionLayer
from crossformer.model.layers.encoder import Encoder
from crossformer.model.layers.decoder import Decoder
from crossformer.model.layers.embedding import PositionEmbedding, ValueEmebedding, DSWEmbedding

from crossformer.utils.metrics import metric

from math import ceil

class Crossformer(nn.Module): 
    def __init__(self, data_dim, in_len, out_len, seg_len, window_size = 4,
                factor=10, model_dim=512, feedforward_dim = 1024, heads_num=8, blocks_num=3, 
                dropout=0.0, baseline = False, **kwargs):
        """
        Args:
            data_dim: int
                The dimension of the input data
            in_len: int
                The length of the input sequence
            out_len: int
                The length of the output sequence
            seg_len: int
                The length of the segment
            window_size: int
                The size of the window
            factor: int
                The factor of the sparse attention
            model_dim: int
                The dimension of the model
            feedforward_dim: int
                The dimension of the feedforward network
            heads_num: int
                The number of heads
            blocks_num: int
                The number of blocks
            dropout: float
                The dropout rate
            baseline: bool
                Whether to use the baseline
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
        self.enc_embedding = ValueEmebedding(seg_len=self.seg_len, model_dim=model_dim)
        self.enc_pos = nn.Parameter(torch.randn(1, data_dim, (self.in_seg_num), model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.encoder = Encoder(blocks_num=blocks_num, model_dim=model_dim, window_size=window_size, depth=1, seg_num=self.in_seg_num, factor=factor, heads_num=heads_num, feedforward_dim=feedforward_dim, dropout=dropout)

        # Decode Embedding & Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.out_seg_num), model_dim))
        self.decoder = Decoder(seg_len=self.seg_len, model_dim=model_dim, heads_num=heads_num, depth=1, feedforward_dim=feedforward_dim, dropout=dropout, out_segment_num=self.out_seg_num, factor=factor)

    def forward(self, x_seq):
        """
        Args:
            x_seq: (batch_size, timeseries_length, timeseries_dim) 
        """
        if (self.baseline):
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_seg_num*self.seg_len != self.in_len):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, (self.seg_len*self.in_seg_num-self.in_len), -1), x_seq), dim=1)
        
        x_seq = self.enc_embedding(x_seq)
        x_seq += self.enc_pos
        x_seq = self.norm(x_seq)

        enc_out = self.encoder(x_seq)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return base + predict_y[:, :self.out_len, :]

class CrossFormer(pl.LightningModule):
    def __init__(self, cfg=None, learning_rate=1e-4, batch=32, **kwargs):
        super(CrossFormer, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        # Create the model
        self.model = Crossformer(**cfg)

        # Training Parameters
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.batch = batch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = self(x) * scale # scale the output (but not correct), need to be fixed
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'train_loss': loss} 

    def validation_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = self(x) * scale # scale the output (but not correct), need to be fixed
        metrics = metric(y_hat, y)
        for key in metrics.keys():
            metrics[f'val_{key}'] = metrics.pop(key)
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics       

    def test_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = self(x) * scale # scale the output (but not correct), need to be fixed
        metrics = metric(y_hat, y)
        for key in metrics.keys():
            metrics[f'test_{key}'] = metrics.pop(key)
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics
    
    def predict_step(self, batch, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        (x, scale, y) = batch
        y_hat = self(x) * scale # scale the output (but not correct), need to be fixed
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
            }
        }