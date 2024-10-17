"""
    Description: Embedding
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class DSWEmbedding(nn.Module):
    def __init__(self, seg_len, model_dim, data_dim, seg_num,):
        super(DSWEmbedding, self).__init__()
        """
            Dimension Segment-wise Embedding Layer.
        """

        self.value_embedding = ValueEmebedding(seg_len=seg_len, model_dim=model_dim)
        self.position_embedding = PositionEmbedding(data_dim=data_dim, seg_num=seg_num, model_dim=model_dim)
        self.norm = nn.LayerNorm(model_dim)
        

    def forward(self, x):
        """
        Reference: h_{i,d} = Ex_{i,d} + Pos_{i,d}, and a layer norm has been applied in this block as well.

        Args:
            x: (batch_size, timeseries_length, timeseries_dim).
            seg_len: The length of the segment.
            model_dim: The dimension of the model.
            data_dim: The dimension of timeseries data.
            seg_num: The number of segments in the timeseries data.
        """
        x_embed = self.value_embedding(x)
        x_embed = self.position_embedding(x_embed)
        x_embed = self.norm(x_embed)

        return x_embed
    

class ValueEmebedding(nn.Module):
    def __init__(self, seg_len, model_dim,):
        super(ValueEmebedding, self).__init__()
        """
        Args:
            seg_len: The length of the segment.
            model_dim: The dimension of the model.
        """
        self.seg_len = seg_len
        
        self.linear = nn.Linear(seg_len, model_dim)

    def forward(self, x):
        """
        Reference: Ex_{i,d}, where E is learnable matrix.

        Args:
            x: (batch_size, timeseries_length, timeseries_dim)
        """
        batch, ts_len, ts_dim = x.size()

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len) 
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) model_dim -> b d seg_num model_dim', b=batch, d=ts_dim)

        return x_embed
    
class PositionEmbedding(nn.Module):
    def __init__(self, data_dim, seg_num, model_dim):
        """
        Args:
            data_dim: The dimension of timeseries data.
            seg_num: The number of segments in the timeseries data.
            model_dim: The dimension of the model.
        """
        super(PositionEmbedding, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, data_dim, seg_num, model_dim))

    def forward(self, x):
        """
        Reference: Pos_{i,d}.

        Args:
            x: (batch_size,  seg_num, seg_len, data_dim)
        """
        x_embed = x + self.positional_embedding
        
        return x_embed