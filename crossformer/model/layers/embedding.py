"""Embeddings.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch.nn as nn
from einops import rearrange


class ValueEmebedding(nn.Module):

    def __init__(
        self,
        seg_len,
        model_dim,
    ):
        """Initializes the ValueEmbedding module.

        Args:
            seg_len (int): The length of the segment.
            model_dim (int): The dimension of the model.
        """
        super(ValueEmebedding, self).__init__()

        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, model_dim)

    def forward(self, x):
        """
        Applies the linear transformation to the input segments.

        Args:
            x (torch.Tensor): Input tensor of shape
            (batch_size, timeseries_length, timeseries_dim).

        Returns:
            torch.Tensor: Transformed tensor of shape
            (batch_size, timeseries_dim, num_segments, model_dim).
        """
        batch, ts_len, ts_dim = x.size()

        x_segment = rearrange(
            x,
            'b (seg_num seg_len) d -> (b d seg_num) seg_len',
            seg_len=self.seg_len,
        )
        x_embed = self.linear(x_segment)
        x_embed = rearrange(
            x_embed,
            '(b d seg_num) model_dim -> b d seg_num model_dim',
            b=batch,
            d=ts_dim,
        )

        return x_embed
