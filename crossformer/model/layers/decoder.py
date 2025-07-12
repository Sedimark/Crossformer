"""Decoder.

Author: Peipei Wu (Paul) - Surrey
Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import torch.nn as nn
from crossformer.model.layers.attention import TwoStageAttentionLayer
from crossformer.model.layers.attention import AttentionLayer
from typing import List


class DecoderLayer(nn.Module):
    """Decoder layer for the TimeSeriesTransformer model."""

    def __init__(
        self,
        seg_len,
        model_dim,
        heads_num,
        feedforward_dim=None,
        dropout=0.1,
        out_segment_num=10,
        factor=10,
    ):
        super(DecoderLayer, self).__init__()

        self.self_attention = TwoStageAttentionLayer(
            seg_num=out_segment_num,
            factor=factor,
            model_dim=model_dim,
            heads_num=heads_num,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.cross_attention = AttentionLayer(
            model_dim, heads_num, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.linear_predict = nn.Linear(model_dim, seg_len)

    def forward(self, x, memory):
        batch, data_dim, out_seg_num, model_dim = x.shape
        x = self.self_attention(x)
        x = x.contiguous().view(batch * data_dim, out_seg_num, model_dim)

        _, _, in_seg_num, _ = memory.shape
        memory = memory.contiguous().view(
            batch * data_dim, in_seg_num, model_dim
        )

        x_decode = self.cross_attention(x, memory, memory)
        x_decode = x + self.dropout(x_decode)
        y = x = self.norm_1(x_decode)
        dec_out = self.norm_2(y + x)

        decode_seg_num = dec_out.shape[1]
        dec_out = dec_out.contiguous().view(
            batch, data_dim, decode_seg_num, model_dim
        )

        layer_predict = self.linear_predict(dec_out)
        b, out_d, seg_num, seg_len = layer_predict.shape
        layer_predict = layer_predict.contiguous().view(
            b, out_d * seg_num, seg_len
        )

        return dec_out, layer_predict


class Decoder(nn.Module):
    """Decoder for the TimeSeriesTransformer model."""

    def __init__(
        self,
        seg_len,
        model_dim,
        heads_num,
        depth,
        feedforward_dim=None,
        dropout=0.1,
        out_segment_num=10,
        factor=10,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    seg_len,
                    model_dim,
                    heads_num,
                    feedforward_dim,
                    dropout,
                    out_segment_num,
                    factor,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, memory: List[torch.Tensor]):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.layers:
            memory_enc = memory[i]
            x, layer_predict = layer(x, memory_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict += layer_predict

            i += 1

        b, total_seg, seg_len = final_predict.shape
        seg_num = total_seg // ts_d
        final_predict = final_predict.view(b, ts_d, seg_num, seg_len)
        final_predict = final_predict.permute(0, 2, 3, 1)
        final_predict = final_predict.reshape(b, seg_num * seg_len, ts_d)
        return final_predict
