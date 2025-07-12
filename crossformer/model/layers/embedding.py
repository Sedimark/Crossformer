"""Embeddings.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch.nn as nn


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
        seg_num = ts_len // self.seg_len

        # (b, t, d) -> (b, seg_num, seg_len, d)
        x = x.view(batch, seg_num, self.seg_len, ts_dim)
        # (b, seg_num, seg_len, d) -> (b, d, seg_num, seg_len)
        x = x.permute(0, 3, 1, 2).contiguous()
        # (b, d, seg_num, seg_len) -> (b * d * seg_num, seg_len)
        x_segment = x.view(batch * ts_dim * seg_num, self.seg_len)

        x_embed = self.linear(x_segment)

        # (b * d * seg_num, model_dim) -> (b, d, seg_num, model_dim)
        x_embed = x_embed.view(batch, ts_dim, seg_num, -1)

        return x_embed
