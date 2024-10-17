"""
    Description: Decoder
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from crossformer.model.layers.embedding import PositionEmbedding
from crossformer.model.layers.attention import TwoStageAttentionLayer, AttentionLayer

class DecoderLayer(nn.Module):
    r"""
        Decoder layer for the TimeSeriesTransformer model.
    """
    def __init__(self, seg_len, model_dim, heads_num, feedforward_dim=None, dropout=0.1, out_segment_num=10, factor=10):
        r""""
        Args:
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attention = TwoStageAttentionLayer(seg_num=out_segment_num, factor=factor, model_dim=model_dim, heads_num=heads_num, feedforward_dim=feedforward_dim, dropout=dropout)
        self.cross_attention = AttentionLayer(model_dim, heads_num, dropout=dropout)

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim),
                                 nn.GELU(),
                                 nn.Linear(model_dim, model_dim))
        
        self.linear_predict = nn.Linear(model_dim, seg_len)


    def forward(self, x, memory):
        r"""
        x: the output of the last decoder layer.
        memory: the output of the corresponding encoder layer.

        Args:
        x: (batch_size, data_dim, seg_num, model_dim)
        memory: (batch_size, data_dim, seg_num, model_dim)
        """

        batch_size = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'batch data_dim out_seg_num model_dim -> (batch data_dim) out_seg_num model_dim')

        memory = rearrange(memory, 'batch data_dim in_seg_num model_dim -> (batch data_dim) in_seg_num model_dim')
        x_decode = self.cross_attention(x, memory, memory)
        x_decode = x + self.dropout(x_decode)
        y = x = self.norm_1(x_decode)
        dec_out = self.norm_2(y + x)

        dec_out = rearrange(dec_out, '(batch data_dim) decode_seg_num model_dim -> batch data_dim decode_seg_num model_dim', batch=batch_size)
        layer_predict = self.linear_predict(dec_out)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len', b=batch_size)

        return dec_out, layer_predict
    

class Decoder(nn.Module):
    r"""
        Decoder for the TimeSeriesTransformer model.
    """
    def __init__(self, seg_len, model_dim, heads_num, depth, feedforward_dim=None, dropout=0.1, out_segment_num=10, factor=10):
        r""""
        Args:
        """
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([DecoderLayer(seg_len, model_dim, heads_num, feedforward_dim, dropout, out_segment_num, factor) for _ in range(depth)])

    def forward(self, x, memory):
        r"""
        x: the output of the encoder.
        memory: the output of the encoder.

        Args:
        x: (batch_size, data_dim, seg_num, model_dim)
        memory: (batch_size, data_dim, seg_num, model_dim)
        """

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
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)
        return final_predict