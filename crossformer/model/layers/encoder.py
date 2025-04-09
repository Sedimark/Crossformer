"""Encoder.

    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import torch.nn as nn
from crossformer.model.layers.attention import TwoStageAttentionLayer

from math import ceil


class SegmentMerging(nn.Module):
    """Segment Merging Layer for Crossformer."""

    def __init__(self, model_dim, window_size):
        """
        Initialize the Segment Merging Layer.

        Args:
            model_dim (int): The dimension of the model.
            window_size (int): The window size for segment merging.
        """
        super(SegmentMerging, self).__init__()

        self.model_dim = model_dim
        self.window_size = window_size
        self.linear = nn.Linear(window_size * model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim * window_size)

    def forward(self, x):
        """
        Forward pass for Segment Merging Layer.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, data_dim, seg_num, model_dim).

        Returns:
            torch.Tensor: Output tensor after segment merging.
        """
        batch_size, data_dim, seg_num, model_dim = x.shape
        pad_num = seg_num % self.window_size
        if pad_num != 0:
            pad_num = self.window_size - pad_num
            x = torch.cat(
                (x, x[:, :, -pad_num:, :]), dim=-2
            )  # (batch_size, data_dim, seg_num + pad_num, model_dim)

        segments = []
        # change to shape
        # (batch_size, data_dim, seg_num//window_size, model_dim)
        for i in range(self.window_size):
            segments.append(x[:, :, i :: self.window_size, :])  # noqa: E203
        # change to shape
        # (batch_size, data_dim, seg_num//window_size, window_size * model_dim)
        x = torch.cat(segments, -1)

        x = self.norm(x)
        x = self.linear(x)

        return x


class Blocks(nn.Module):
    """
    Blocks for Crossformer's Encoder.
    """

    def __init__(
        self,
        model_dim,
        window_size,
        depth,
        seg_num,
        factor,
        heads_num,
        feedforward_dim=None,
        dropout=0.1,
    ):
        """
        Initialize the Blocks for Crossformer's Encoder.

        Args:
            model_dim (int): The dimension of the model.
            window_size (int): The window size for segment merging.
            depth (int): The number of encoding layers.
            seg_num (int): The number of segments.
            factor (int): The factor for the attention mechanism.
            heads_num (int): The number of attention heads.
            feedforward_dim (int, optional): The dimension of the feedforward
                                             network. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(Blocks, self).__init__()

        if window_size > 1:
            self.merge = SegmentMerging(model_dim, window_size)
        else:
            self.merge = None

        self.encode_layer = nn.ModuleList()

        for i in range(depth):
            self.encode_layer.append(
                TwoStageAttentionLayer(
                    seg_num=seg_num,
                    factor=factor,
                    model_dim=model_dim,
                    heads_num=heads_num,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        """
        Forward pass for Blocks.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, data_dim, seg_num, model_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the encoding
                          layers.
        """

        _, data_dim, _, _ = x.shape

        if self.merge is not None:
            x = self.merge(x)

        for layer in self.encode_layer:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """Encoder for Crossformer."""

    def __init__(
        self,
        blocks_num,
        model_dim,
        window_size,
        depth,
        seg_num,
        factor,
        heads_num,
        feedforward_dim=None,
        dropout=0.1,
    ):
        """
        Initialize the Encoder for Crossformer.

        Args:
            blocks_num (int): The number of blocks in the encoder.
            model_dim (int): The dimension of the model.
            window_size (int): The window size for segment merging.
            depth (int): The number of encoding layers in each block.
            seg_num (int): The number of segments.
            factor (int): The factor for the attention mechanism.
            heads_num (int): The number of attention heads.
            feedforward_dim (int, optional): The dimension of the feedforward
                             network. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList()

        self.encoder.append(
            Blocks(
                model_dim,
                1,
                depth,
                seg_num,
                factor,
                heads_num,
                feedforward_dim,
                dropout=dropout,
            )
        )  # first layer with window_size = 1
        for i in range(1, blocks_num):
            self.encoder.append(
                Blocks(
                    model_dim,
                    window_size,
                    depth,
                    ceil(seg_num / window_size**i),
                    factor,
                    heads_num,
                    feedforward_dim,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        """
        Forward pass for Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape
                      (batch_size, data_dim, seg_num, model_dim).

        Returns:
            List[torch.Tensor]: List of output tensors after each block.
        """

        encode_x = []
        encode_x.append(x)

        for layer in self.encoder:
            x = layer(x)
            encode_x.append(x)

        return encode_x
