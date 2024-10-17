"""
    Description: Attention Layer
    Author: Peipei Wu (Paul) - Surrey
    Maintainer: Peipei Wu (Paul) - Surrey
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k ,v):
        B, L, H, E = q.shape # Batch, Length, Heads, Embedding
        _, S, _, D = k.shape # Batch, Sequence, Heads, Embedding

        scale = (E ** -0.5)

        scores = torch.einsum('b l h e, b s h d -> b h l s', q, k)
        attention = self.dropout(F.softmax(scores*scale,dim=-1))
        out = torch.einsum('b h l s, b s h d -> b l h d', attention, v)
        return out.contiguous()
    

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, heads_num, dropout=0.1):
        super().__init__()

        k_dim = model_dim // heads_num
        v_dim = model_dim // heads_num

        self.attention = Attention(dropout=dropout)

        self.q_proj = nn.Linear(model_dim, k_dim * heads_num)
        self.k_proj = nn.Linear(model_dim, k_dim * heads_num)
        self.v_proj = nn.Linear(model_dim, v_dim * heads_num)

        self.out_proj = nn.Linear(model_dim, model_dim)
        self.heads_num = heads_num

    def forward(self, q, k, v):
        B, L, E = q.shape
        _, S, _ = k.shape
       
        q = self.q_proj(q).view(B, L, self.heads_num, -1)
        k = self.k_proj(k).view(B, S, self.heads_num, -1)
        v = self.v_proj(v).view(B, S, self.heads_num, -1)

        out = self.attention(q, k, v)

        out = rearrange(out, 'b l h d -> b l (h d)')
        out = self.out_proj(out)

        return out

class TwoStageAttentionLayer(nn.Module):
    def __init__(self, seg_num, factor, model_dim, heads_num, feedforward_dim=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        feedforward_dim = feedforward_dim or 4 * model_dim
        self.time_attention = AttentionLayer(model_dim, heads_num, dropout=dropout)
        self.dim_sender = AttentionLayer(model_dim, heads_num, dropout=dropout) 
        self.dim_receiver = AttentionLayer(model_dim, heads_num, dropout=dropout) 
        self.router = nn.Parameter(torch.randn(seg_num, factor, model_dim))
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.norm4 = nn.LayerNorm(model_dim)

        self.MLP1 = nn.Sequential(nn.Linear(model_dim, feedforward_dim),
                                  nn.GELU(),
                                  nn.Linear(feedforward_dim, model_dim))
        self.MLP2 = nn.Sequential(nn.Linear(model_dim, feedforward_dim),
                                  nn.GELU(),
                                  nn.Linear(feedforward_dim, model_dim))

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x, 'b data_dim seg_num model_dim -> (b data_dim) seg_num model_dim')
        
        # Cross Time Stage
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage
        dim_send = rearrange(dim_in, '(b data_dim) seg_num model_dim -> (b seg_num)  data_dim  model_dim', b=batch)
        batch_router = repeat(self.router, 'seg_num factor model_dim -> (repeat seg_num)  factor  model_dim', repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num)  data_dim model_dim -> b data_dim seg_num model_dim', b=batch)

        return final_out