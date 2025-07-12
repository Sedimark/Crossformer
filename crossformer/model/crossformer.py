from math import ceil
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
from crossformer.model.layers.encoder import Encoder
from crossformer.model.layers.decoder import Decoder
from crossformer.model.layers.embedding import ValueEmebedding
from crossformer.utils.metrics import metric, hybrid_loss


class Crossformer(nn.Module):

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
        super(Crossformer, self).__init__()

        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = window_size
        self.baseline = baseline

        self.in_seg_num = ceil(1.0 * in_len / seg_len)
        self.out_seg_num = ceil(1.0 * out_len / seg_len)

        self.enc_embedding = ValueEmebedding(
            seg_len=self.seg_len, model_dim=model_dim
        )
        self.enc_pos = nn.Parameter(
            torch.randn(1, data_dim, self.in_seg_num, model_dim)
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

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, data_dim, self.out_seg_num, model_dim)
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
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = torch.zeros(
                x_seq.size(0),
                1,
                x_seq.size(2),
                device=x_seq.device,
                dtype=x_seq.dtype,
            )

        batch_size = x_seq.shape[0]

        if self.in_seg_num * self.seg_len != self.in_len:
            pad_len = self.seg_len * self.in_seg_num - self.in_len
            x_seq = torch.cat(
                [x_seq[:, :1, :].expand(-1, pad_len, -1), x_seq], dim=1
            )

        x_seq = self.enc_embedding(x_seq)
        x_seq = x_seq + self.enc_pos
        x_seq = self.norm(x_seq)

        enc_out = self.encoder(x_seq)

        # (1, data_dim, seg_num, model_dim) -> (batch_size, data_dim, seg_num, model_dim)
        dec_in = self.dec_pos_embedding.expand(batch_size, -1, -1, -1)

        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, : self.out_len, :]


class CrossFormer(LightningModule):

    def __init__(self, cfg=None, learning_rate=1e-4, batch=32, **kwargs):
        super(CrossFormer, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = Crossformer(**cfg)
        self.loss = hybrid_loss
        self.learning_rate = learning_rate
        self.batch = batch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = (
            self(x) * scale.unsqueeze(1) if scale._is_zerotensor() else self(x)
        )
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = (
            self(x) * scale.unsqueeze(1) if scale._is_zerotensor() else self(x)
        )
        metrics = metric(y_hat, y)
        metrics = {f"val_{key}": value for key, value in metrics.items()}
        self.log_dict(
            metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return metrics

    def test_step(self, batch, batch_idx):
        (x, scale, y) = batch
        y_hat = (
            self(x) * scale.unsqueeze(1) if scale._is_zerotensor() else self(x)
        )
        metrics = metric(y_hat, y)
        metrics = {f"test_{key}": value for key, value in metrics.items()}
        self.log_dict(
            metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        return metrics

    def predict_step(self, batch, *args, **kwargs):
        (x, scale, y) = batch
        y_hat = (
            self(x) * scale.unsqueeze(1) if scale._is_zerotensor() else self(x)
        )
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.1 ** (epoch // 25)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_SCORE",
                "interval": "epoch",
                "frequency": 1,
            },
        }
