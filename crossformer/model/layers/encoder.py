"""
    Description: Encoder
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from crossformer.model.layers.attention import TwoStageAttentionLayer

from math import ceil


class SegmentMerging(nn.Module):
    r"""
    Segment Merging Layer for Crossformer.
    Args:
    """
    def __init__(self, model_dim, window_size):
        super(SegmentMerging, self).__init__()

        self.model_dim = model_dim
        self.window_size = window_size
        self.linear = nn.Linear(window_size * model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim*window_size)

    def forward(self, x):
        r""""
        Args:
        x: (batch_size, data_dim, seg_num, model_dim)
        """
        batch_size, data_dim, seg_num, model_dim = x.shape
        pad_num = seg_num % self.window_size
        if pad_num != 0:
            pad_num = self.window_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2) # (batch_size, data_dim, seg_num + pad_num, model_dim)

        segments = []
        for i in range(self.window_size):
            segments.append(x[:, :, i::self.window_size, :]) # (batch_size, data_dim, seg_num//window_size, model_dim)
        x = torch.cat(segments, -1) # (batch_size, data_dim, seg_num//window_size, window_size * model_dim)

        x = self.norm(x)
        x = self.linear(x)

        return x
    
class Blocks(nn.Module):
    r"""
        Blocks for Crossformer's Encoder.

        Args:
        model_dim: The dimension of the model.
        window_size: The window size for segment merging.
        depth: The depth of the encoder (number of TSA layers).
        seg_num: The number of segments in the timeseries data.
        factor: The factor for routers.
        heads_num: The number of heads in the multi-head attention.
        feedforward_dim: The dimension of the feedforward layer.
        dropout: The dropout rate.
    """
    def __init__(self, model_dim, window_size, depth, seg_num, factor, heads_num, feedforward_dim=None, dropout=0.1):
        super(Blocks, self).__init__()

        if window_size > 1:
            self.merge = SegmentMerging(model_dim, window_size)
        else:
            self.merge = None

        self.encode_layer = nn.ModuleList()

        for i in range(depth):
            self.encode_layer.append(TwoStageAttentionLayer(seg_num=seg_num, factor=factor, model_dim=model_dim, heads_num=heads_num, feedforward_dim=feedforward_dim, dropout=dropout))


    def forward(self, x):
        r"""
        Args:
        x: (batch_size, data_dim, seg_num, model_dim)
        """

        _, data_dim, _, _ = x.shape

        if self.merge is not None:
            x = self.merge(x)
        
        for layer in self.encode_layer:
            x = layer(x)
        
        return x
    


class Encoder(nn.Module):
    r"""
    Encoder for Crossformer.
    """
    def __init__(self, blocks_num, model_dim, window_size, depth, seg_num, factor, heads_num, feedforward_dim=None, dropout=0.1):
        super(Encoder, self).__init__()


        self.encoder = nn.ModuleList()

        self.encoder.append(Blocks(model_dim, 1, depth, seg_num, factor, heads_num, feedforward_dim, dropout=dropout)) # first layer with window_size = 1
        for i in range(1, blocks_num):
            self.encoder.append(Blocks(model_dim, window_size, depth, ceil(seg_num/window_size**i), factor, heads_num, feedforward_dim, dropout=dropout))


    def forward(self, x):
        r"""
        Args:
        x: (batch_size, data_dim, seg_num, model_dim)
        """

        encode_x = []
        encode_x.append(x)

        for layer in self.encoder:
            x = layer(x)
            encode_x.append(x)
        
        return encode_x