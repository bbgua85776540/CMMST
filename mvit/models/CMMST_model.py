# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from .build import MODEL_REGISTRY
from mvit.utils.misc import validate_checkpoint_wrapper_import
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None
from einops import rearrange
import math
from functools import partial



class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """
    def __init__(
            self,
            dim_in=1,
            dim_out=12,
            kernel=(18, 32),
            stride=(18, 32),
            padding=(1, 1),
            patch_len = 5
    ):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.MaxPool2d(
            kernel_size=kernel,
            stride=stride,
            # padding=padding,
        )
        self.ln = nn.Linear(dim_out*8*8, 8)

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x  # torch.Size([2, 3136, 96])





@MODEL_REGISTRY.register()
class CMMST(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.MVIT.EMBED_DIM
        self.lookback = cfg.fps
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.emb = nn.Linear(2, cfg.MVIT.EMBED_DIM)  ###
        # self.emb = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, self.lookback, cfg.MVIT.EMBED_DIM))
        self.dropout = nn.Dropout(0.3)
        self.decoder1 = nn.Linear(embed_dim, 2)
        self.decoder2 = nn.Linear(2, 64)
        # self.decoder3 = nn.Conv2d(in_channels=90, out_channels=30, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # self.decoder4 = nn.Conv2d(in_channels=30, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.process_num = cfg.process_frame_nums

        normlayer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = normlayer(embed_dim)

        self.norm_trj = nn.LayerNorm(2)
        # self.norm_pre = nn.LayerNorm(cfg.MVIT.EMBED_DIM)
        self.linear = nn.Linear(cfg.fps, 1)

        # self.emb = nn.Embedding(cfg.MVIT.EMBED_DIM, 64)
        # self.user_embedding = torch.nn.Embedding(48, 2)
        # self.sig = nn.Sigmoid()
        # self.patch_emb = PatchEmbed(dim_out=1)

        self.output_softmax = nn.Softmax(dim=-1)
        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
        self.zero_decay_pos_cls = cfg.MVIT.ZERO_DECAY_POS_CLS
        self.apply(self._init_weights)
        # trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            # names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]
        return names

    def forward(self, x):

        headmaps, salmaps = x

        B = headmaps.shape[0]

        # user_emb = self.user_embedding(user)
        # user_vector = user_emb.unsqueeze(1).expand(B, self.process_num, 2)

        headmaps_emb = self.emb(headmaps)

        x = headmaps_emb

        x = x + self.pos_embed
        x = self.dropout(x)
        # x = self.norm_pre(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        # x = torch.flatten(x,1)
        trj = self.decoder1(x)
        # trj = self.norm_trj(trj)
        # x = self.decoder2(trj)
        # output = self.output_softmax(x)
        return None, trj



#### positional encoding ####
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

