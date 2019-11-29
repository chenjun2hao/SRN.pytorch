# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.pos_embed = nn.Embedding(cfg.p_dim, cfg.dim) # position embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)    # (S,) -> (B, S)

        e = x + self.pos_embed(pos) 
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg, n_layers):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, x, mask):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, mask)
        return h


class Parallel_Attention(nn.Module):
    ''' the Parallel Attention Module for 2D attention
        reference the origin paper: https://arxiv.org/abs/1906.05708
    '''
    def __init__(self, cfg):
        super().__init__()
        self.atten_w1 = nn.Linear(cfg.dim_c, cfg.dim_c)
        self.atten_w2 = nn.Linear(cfg.dim_c, cfg.max_vocab_size)
        self.activ_fn = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(0.1)

    def forward(self, origin_I, bert_out, mask=None):
        bert_out = self.activ_fn(self.drop(self.atten_w1(bert_out)))
        atten_w = self.soft(self.atten_w2(bert_out))                  # b*200*94
        x = torch.bmm(origin_I.transpose(1,2), atten_w)               # b*512*94
        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head=8, d_k=64, d_model=128, max_vocab_size=94, dropout=0.1):
        ''' d_k: the attention dim
            d_model: the encoder output feature
            max_vocab_size: the output maxium length of sequence
        '''
        super(MultiHeadAttention, self).__init__()

        self.n_head, self.d_k = n_head, d_k
        self.temperature = np.power(d_k, 0.5)
        self.max_vocab_size = max_vocab_size

        self.w_encoder = nn.Linear(d_model, n_head * d_k)
        self.w_atten = nn.Linear(d_model, n_head * max_vocab_size)
        self.w_out = nn.Linear(n_head * d_k, d_model)
        self.activ_fn = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)                                # at the d_in dimension
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.w_encoder.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_atten.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.xavier_normal_(self.w_out.weight)


    def forward(self, encoder_feature, bert_out, mask=None):

        d_k, n_head, max_vocab_size = self.d_k, self.n_head, self.max_vocab_size

        sz_b, d_in, _ = encoder_feature.size()
        
        # 原始特征
        encoder_feature = encoder_feature.view(sz_b, d_in, n_head, d_k)
        encoder_feature = encoder_feature.permute(2, 0, 1, 3).contiguous().view(-1, d_in, d_k)        # 32*200*64
        
        # 求解权值
        alpha = self.activ_fn(self.dropout(self.w_encoder(bert_out)))
        alpha = self.w_atten(alpha).view(sz_b, d_in, n_head, max_vocab_size)            # 4*200*8*94
        alpha = alpha.permute(2, 0, 1, 3).contiguous().view(-1, d_in, max_vocab_size)   # 32*200*94
        alpha = alpha / self.temperature
        alpha = self.dropout(self.softmax(alpha))                         # 32*200*94
        
        # 输出部分
        output = torch.bmm(encoder_feature.transpose(1,2), alpha)         # 32*64*94
        output = output.view(n_head, sz_b, d_k, max_vocab_size)
        output = output.permute(1, 3, 0, 2).contiguous().view(sz_b, max_vocab_size, -1)  # 4*94*512
        output = self.dropout(self.w_out(output))
        output = output.transpose(1,2)

        return output


class Two_Stage_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_w = nn.Linear(cfg.dim_c, cfg.len_alphabet)
        self.relation_attention = Transformer(cfg, cfg.decoder_atten_layers)
        self.out_w1 = nn.Linear(cfg.dim_c, cfg.len_alphabet)

    def forward(self, x):
        x1 = self.out_w(x)
        x2 = self.relation_attention(x, mask=None)
        x2 = self.out_w1(x2)                    # 两个分支的输出部分采用不同的网络

        return x1, x2


class Bert_Ocr(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg, cfg.attention_layers)
        self.attention = Parallel_Attention(cfg)
#         self.attention = MultiHeadAttention(d_model=cfg.dim, max_vocab_size=cfg.max_vocab_size)
        self.decoder = Two_Stage_Decoder(cfg)

    def forward(self, encoder_feature, mask=None):
        bert_out = self.transformer(encoder_feature, mask)                 # 做一个self_attention//4*200*512
        glimpses = self.attention(encoder_feature, bert_out, mask)         # 原始序列和目标序列的转化//4*512*94
        res = self.decoder(glimpses.transpose(1,2))
        return res


class Config(object):
    '''参数设置'''
    """ Relation Attention Module """
    p_drop_attn = 0.1
    p_drop_hidden = 0.1
    dim = 512                       # the encode output feature
    attention_layers = 2                    # the layers of transformer
    n_heads = 8
    dim_ff = 1024 * 2                       # 位置前向传播的隐含层维度

    ''' Parallel Attention Module '''
    dim_c = dim
    max_vocab_size = 26             # 一张图片含有字符的最大长度

    """ Two-stage Decoder """
    len_alphabet = 39               # 字符类别数量
    decoder_atten_layers = 2


def numel(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':

    cfg = Config()
    mask = None
    x = torch.randn(4, 200, cfg.dim)
    net = Bert_Ocr(cfg)
    res1, res2 = net(x, mask)
    print(res1.shape, res2.shape)
    print('参数总量为:', numel(net))
