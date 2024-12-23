# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial
from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
from mvit.models.attention import MultiScaleBlock, MultiScaleFuseBlock
from mvit.models.common import round_width
from mvit.utils.misc import validate_checkpoint_wrapper_import
from torch.nn.init import trunc_normal_
import numpy as np
from .build import MODEL_REGISTRY
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None
flash_attn_unpadded_func = None





class att_upsample(nn.Module):
    def __init__(
            self,
            x_15_trj_len=2,
            x_10_len=6,
            x_15_len=3,
            x_10_trj_len = 3
    ):
        super().__init__()
        # self.Ln = nn.Linear(x_10_len, x_len)
        # self.Ln2 = nn.Linear(x_10_len, x_10_len)
        self.Ln22 = nn.Linear(x_10_len, x_15_len)
        # self.Ln3 = nn.Linear(x_10_trj_len, x_10_trj_len)
        self.Ln33 = nn.Linear(x_10_trj_len, x_15_trj_len)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_10, x_10_trj, x_15, x_15_trj, thw_10, thw_10_trj, thw_15, thw_15_trj):
        # B, C = x_10.shape[0], x_10.shape[2]

        # x_10 = rearrange(x_10, 'b (t n) c -> b t n c', t=thw_10[0])
        # x_10_trj = rearrange(x_10_trj, 'b (t n) c -> b t n c', t=thw_10_trj[0])


        # x_15 = rearrange(x_15, 'b (t n) c -> b c n t', t=thw_15[0])
        # x_15 = self.Ln2(x_15)
        # x_15_trj = rearrange(x_15_trj, 'b (t n) c -> b c n t', t=thw_15_trj[0])
        # x_15_trj = self.Ln3(x_15_trj)
        # x_15 = rearrange(x_15, 'b c n t -> b t n c')
        # x_15_trj = rearrange(x_15_trj, 'b c n t -> b t n c')

        x_10 = rearrange(x_10, 'b (t n) c -> b c n t', t=thw_10[0])
        x_10_trj = rearrange(x_10_trj, 'b (t n) c -> b c n t', t=thw_10_trj[0])

        x_10_fus = torch.cat((x_10, x_10_trj), dim=-1)
        T_10_fus = x_10_fus.shape[-1]
        x_10_fus = rearrange(x_10_fus, 'b c n t -> b (t n) c')

        # x_10 = self.Ln2(x_10)
        x_10 = self.Ln22(x_10)

        # x_10_trj = self.Ln3(x_10_trj)
        x_10_trj = self.Ln33(x_10_trj)

        x_10 = rearrange(x_10, 'b c n t -> b t n c')
        x_10_trj = rearrange(x_10_trj, 'b c n t -> b t n c')

        x_15 = rearrange(x_15, 'b (t n) c -> b t n c', t=thw_15[0])
        x_15_trj = rearrange(x_15_trj, 'b (t n) c -> b t n c', t=thw_15_trj[0])
        x_15 = x_10 * x_15
        x_15_trj = x_10_trj * x_15_trj

        x_15_fus = torch.cat((x_15,x_15_trj), dim=1)
        T_fus = x_15_fus.shape[1]
        x_15 = rearrange(x_15, 'b t n c -> b (t n) c')
        x_15_trj = rearrange(x_15_trj, 'b t n c -> b (t n) c')
        x_15_fus = rearrange(x_15_fus, 'b t n c -> b (t n) c')



        return x_15, x_15_trj, x_15_fus, T_fus, x_10_fus, T_10_fus



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 use_checkpoint=False, T=8, config=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers
        if use_checkpoint and checkpoint_wrapper:
            self.resblocks = nn.ModuleList([
                checkpoint_wrapper(TemporalBlock(width, heads, attn_mask, droppath[i], config=config))
                for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([TemporalBlock(width, heads, attn_mask, droppath[i], config=config)
                                            for i in range(layers)])

    def forward(self, x: torch.Tensor, THW):
        # L, NT, C = x.shape
        # T, H, W = THW
        # N = NT // T
        # H = W = int((L - 1) ** 0.5)
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, THW)
        return x


class LocalTemporal(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_bottleneck = d_model // 2

        self.ln = LayerNorm(d_model)
        self.down_proj = nn.Conv3d(d_model, d_bottleneck, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(d_bottleneck, d_bottleneck, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                              groups=d_bottleneck)
        self.up_proj = nn.Conv3d(d_bottleneck, d_model, kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.up_proj.weight, 0)
        nn.init.constant_(self.up_proj.bias, 0)

        self.activation = QuickGELU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.down_proj(x)
        x = self.activation(self.conv(x))
        x = self.up_proj(x)

        return x


class MVIT_LocalTemporal(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_bottleneck = d_model // 2

        self.ln = LayerNorm(d_model)
        self.down_proj = nn.Conv3d(d_model, d_bottleneck, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv3d(d_bottleneck, d_bottleneck, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                              groups=d_bottleneck)
        self.up_proj = nn.Conv3d(d_bottleneck, d_model, kernel_size=1, stride=1, padding=0)

        nn.init.constant_(self.up_proj.weight, 0)
        nn.init.constant_(self.up_proj.bias, 0)

        self.activation = QuickGELU()

    def forward(self, x):
        # b c t h w
        x = x.permute(0, 2, 3, 4, 1)     # b t h w c
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)     # b c t h w
        x = self.down_proj(x)
        x = self.activation(self.conv(x))
        x = self.up_proj(x)

        return x


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        # assert flash_attn_unpadded_func is not None, "FlashAttention is not installed."
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
                         kdim, vdim, batch_first, device, dtype)

    def attention(
            self,
            q, k, v,
            batch_size=1,
            seqlen=77,
            softmax_scale=None,
            attention_dropout=0.0,
            causal=False,
            cu_seqlens=None,
            max_s=None,
            need_weights=False
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value. each of (B*S, H, D)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda

        if cu_seqlens is None:
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
            output = flash_attn_unpadded_func(
                q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, attention_dropout,
                softmax_scale=softmax_scale, causal=causal
            )

        return output

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        seqlen, batch_size, embed_dim = query.shape

        # in-projection and rearrange `s b (3 h d) -> s b (h d) -> (b s) h d`
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        k = k.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)
        v = v.transpose(0, 1).contiguous().view(batch_size * seqlen, self.num_heads, self.head_dim)

        # flash attention (use causal mask)
        causal = attn_mask is not None
        attn_output = self.attention(q, k, v, batch_size, seqlen, causal=causal)

        # out-projection
        # `(b s) h d -> s b (h d)`
        attn_output = attn_output.contiguous().view(batch_size, seqlen, self.num_heads, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seqlen, batch_size, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, None


class TemporalBlock(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None, drop_path=0.0,
            dw_reduction=1.5, config=None
    ):
        super().__init__()

        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.config = config
        self.lmhra1 = LocalTemporal(d_model)
        self.lmhra2 = LocalTemporal(d_model)

        # spatial
        if flash_attn_unpadded_func:
            print("Using Flash Attention")
            self.attn = MultiheadAttention(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, tmp_x, THW, use_checkpoint=False):
        # x: 1+HW, NT, C
        T, H, W = THW
        L, NT, C = tmp_x.shape
        N = NT // T
        # H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()  # (N C T H W)
        tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)

        # MHSA
        tmp_x = tmp_x + self.drop_path(self.attention(self.ln_1(tmp_x)))
        if True:
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        # FFN
        tmp_x = tmp_x + self.drop_path(self.mlp(self.ln_2(tmp_x)))
        return tmp_x


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
    ):
        super().__init__()

        self.proj = nn.MaxPool3d(
            kernel_size=kernel,
            stride=stride,
            # padding=padding,
        )
        self.ln = nn.Linear(1, dim_out)

    def forward(self, x):
        # B, T = x.shape[0], x.shape[1]
        # x = rearrange(x, 'b t c h w -> (b t) c h w')
        # x = x.transpose(1, 2)
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b t h w c', b=B, t=T)
        # print('x.shape=', x.shape)  # x.shape= torch.Size([2, 96, 56, 56])
        # B C H W -> B HW C
        x = self.ln(x)
        x = rearrange(x, 'b t h w c -> b c t h w', b=B, t=T)
        return x, [B, C, T, H, W ]  # torch.Size([2, 3136, 96])


class CubeEmbed(nn.Module):
    """
    CubeEmbed.
    """
    def __init__(
            self,
            dim_in=1,
            dim_out=200,
            # fps_stride=5,
            kernel=(5, 5, 8),
            stride=(5, 5, 8),
            padding=(0, 1, 1),
    ):
        super().__init__()
        self.proj1 = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.activate1 = nn.ReLU()
        self.proj2 = nn.Conv3d(
            in_channels=dim_out,
            out_channels=dim_out,
            kernel_size=(5, 3, 3),
            stride=(5, 2, 2),
            padding=(0, 1, 1),
        )
        self.activate2 = nn.ReLU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.proj1(x)
        x = self.activate1(x)
        x = self.proj2(x)
        x = self.activate2(x)
        # B C H W -> B HW C
        # return x.flatten(3).transpose(1, 2), x.shape
        return x, x.shape

class CubeEmbed_mul(nn.Module):
    """
    CubeEmbed.
    """

    def __init__(
            self,
            dim_in=1,
            dim_out=200,
            # fps_stride=5,
            kernel=(5, 5, 8),
            stride=(5, 5, 8),
            padding=(0, 1, 1),
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.transpose(1, 2)    #
        x = self.proj(x)
        # x = self.relu(x)
        # B C H W -> B HW C
        # return x.flatten(3).transpose(1, 2), x.shape
        return x, x.shape


class CubeEmbed_sal(nn.Module):
    """
    CubeEmbed.
    """

    def __init__(
            self,
            dim_in=1,
            dim_out=200,
            # fps_stride=5,
            kernel=(5, 5, 8),
            stride=(5, 5, 8),
            padding=(0, 1, 1),
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        # self.proj = nn.AvgPool3d(
        #     kernel_size=kernel,
        #     stride=stride,
        #     padding=padding,
        # )
        self.activate = nn.GELU()


    def forward(self, x):    #   (16, 60, 144, 256, 3)          以前  (16, 60, 1, 144, 256)   -》 (16, 1, 60, 144, 256)
        # x = x.transpose(1, 2)
        # x = rearrange(x, 'b t h w c -> b c t h w')    #   输入改为video
        x = self.proj(x)
        x = self.activate(x)
        # B C H W -> B HW C
        # return x.flatten(3).transpose(1, 2), x.shape
        return x, x.shape


class CubeEmbed_sal_mul(nn.Module):
    """
    CubeEmbed.  继续时间下采样
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=200,
            # fps_stride=5,
            kernel=(5, 5, 8),
            stride=(5, 5, 8),
            padding=(0, 1, 1),
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.activate = nn.GELU()

    def forward(self, x):
        # x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.activate(x)
        # B C H W -> B HW C
        # return x.flatten(3).transpose(1, 2), x.shape
        return x, x.shape

class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            dropout_rate=0.0,
            act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # if not self.training:
        #     x = self.act(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        # Prepare input.
        in_chans = 3
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # MViT params.
        num_heads = cfg.MVIT.NUM_HEADS
        depth = cfg.MVIT.DEPTH
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.zero_decay_pos_cls = cfg.MVIT.ZERO_DECAY_POS_CLS
        self.fps10_stride = 2
        self.fps15_stride = 2
        self.fps = cfg.fps
        self.fps_stride = cfg.fps_stride
        self.use_sal = cfg.use_sal
        self.process_frame_nums = cfg.process_frame_nums
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.relu = nn.ReLU()
        x_len = int(self.process_frame_nums / self.fps_stride)  #
        x_10_len = int((x_len + 1) // self.fps10_stride)
        x_10_trj_len = ((cfg.fps//cfg.fps_stride)+1)//2

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)

        self.embed_flag = 1
        self.attention_type = cfg.attention_type

        self.patch_embed = CubeEmbed(dim_out=cfg.MVIT.EMBED_DIM, kernel=(4, 4, 4), stride=(2, 4, 4),
                                     padding=(1, 0, 0))
        # self.patch_embed_10 = CubeEmbed_mul(dim_in=cfg.MVIT.EMBED_DIM, dim_out=cfg.MVIT.EMBED_DIM, kernel=(3, 5, 3), stride=(self.fps10_stride, 1, 1), padding=(1,0,1))
        # self.patch_embed_15 = CubeEmbed_mul(dim_in=cfg.MVIT.EMBED_DIM, dim_out=cfg.MVIT.EMBED_DIM, kernel=(3, 3, 3),
        #                                     stride=(self.fps10_stride, 1, 1), padding=(1, 1, 1))

        self.patch_embed_sal = CubeEmbed(dim_out=cfg.MVIT.EMBED_DIM, kernel=(4, 4, 4), stride=(2, 4, 4),
                                     padding=(1, 0, 0))
        # self.patch_embed_sal_10 = CubeEmbed_sal_mul(dim_in=cfg.MVIT.EMBED_DIM, dim_out=cfg.MVIT.EMBED_DIM, kernel=(3, 5, 3), stride=(self.fps10_stride, 1, 1), padding=(1,0,1))
        # self.patch_embed_sal_15 = CubeEmbed_sal_mul(dim_in=cfg.MVIT.EMBED_DIM, dim_out=cfg.MVIT.EMBED_DIM, kernel=(3, 3, 3),
        #                                     stride=(self.fps10_stride, 1, 1), padding=(1, 1, 1))
        # self.upsample = att_upsample()

        patch_dims = [
            spatial_size // cfg.MVIT.PATCH_STRIDE[0],
            spatial_size // cfg.MVIT.PATCH_STRIDE[1],
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, cfg.MVIT.DROPPATH_RATE, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        # self.head_token = nn.Parameter(torch.zeros(8, int(cfg.process_frame_nums-cfg.fps), 144, 256))
        if self.use_abs_pos:
            # self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))
            self.pos_embed_trj = nn.Parameter(torch.zeros(1, 576, cfg.MVIT.EMBED_DIM))  # ([2, 10, 40, 928])
            self.pos_embed = nn.Parameter(torch.zeros(1, 576, cfg.MVIT.EMBED_DIM))
            if self.use_sal:
                self.time_embed = nn.Parameter(
                    torch.zeros(1, x_10_len, cfg.MVIT.EMBED_DIM))
                self.time_embed_trj = nn.Parameter(
                    torch.zeros(1, x_10_trj_len, cfg.MVIT.EMBED_DIM))
                self.time_embed_10 = nn.Parameter(
                    torch.zeros(1, int(cfg.process_frame_nums / cfg.fps_stride), cfg.MVIT.EMBED_DIM*2))
                self.time_embed_15 = nn.Parameter(
                    torch.zeros(1, int(cfg.process_frame_nums / cfg.fps_stride), cfg.MVIT.EMBED_DIM*4))
            else:
                # self.time_embed = nn.Parameter(
                #     torch.zeros(1, int(cfg.fps / cfg.fps_stride), cfg.MVIT.EMBED_DIM))
                self.time_embed = nn.Parameter(
                    torch.zeros(1, int(cfg.process_frame_nums / cfg.fps_stride), cfg.MVIT.EMBED_DIM))
                # self.time_embed_10 = nn.Parameter(
                #     torch.zeros(1, int(cfg.process_frame_nums / 10), cfg.MVIT.EMBED_DIM))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv, pool_t, fuse_stride_q, fuse_stride_kv = _prepare_mvit_configs(
            cfg
        )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        self.fuse_blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.MVIT.MLP_RATIO,
                qkv_bias=cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                pool_t=pool_t[i] if len(pool_t) > i else [],
            )

            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            # self.sal_block.append(sal_down_block)
            self.blocks.append(MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.MVIT.MLP_RATIO,
                qkv_bias=cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                pool_t=pool_t[i] if len(pool_t) > i else [],
            ))
            self.fuse_blocks.append(MultiScaleFuseBlock(
                dim=cfg.MVIT.EMBED_DIM*4,
                dim_out=cfg.MVIT.EMBED_DIM*4,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=cfg.MVIT.MLP_RATIO,
                qkv_bias=cfg.MVIT.QKV_BIAS,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=fuse_stride_q[i] if len(fuse_stride_q) > i else [],
                stride_kv=fuse_stride_kv[i] if len(fuse_stride_kv) > i else [],
                mode=cfg.MVIT.MODE,
                has_cls_embed=self.cls_embed_on,
                pool_first=cfg.MVIT.POOL_FIRST,
                rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                pool_t=pool_t[i] if len(pool_t) > i else [],
            ))

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]
            embed_dim = dim_out

        # self.transformer = TemporalTransformer(embed_dim, layers=4, heads=6, droppath=dpr,
        #                                        use_checkpoint=False, config=cfg)
        # self.norm_10 = norm_layer(embed_dim)
        # self.norm_15 = norm_layer(embed_dim)
        # self.norm_output = norm_layer(64)
        self.output_softmax = nn.Softmax(dim=-1)
        # self.decoder_trj = nn.Conv3d(out_channels=5, in_channels=10, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1))
        # self.decoder_trj2 = nn.Conv3d(out_channels=1, in_channels=5, kernel_size=(3, 3, 3), stride=(3, 1, 1),
        #                              padding=(0, 1, 1))

        # x_10_len = x_len
        # x_10_trj_len = (cfg.fps // cfg.fps_stride)
        self.decoder_emb1 = nn.Linear(embed_dim, embed_dim//2)
        self.decoder_emb2 = nn.Linear(embed_dim//2, 1)
        # self.decoder2 = nn.Linear(5, 5)
        # self.local_temp = MVIT_LocalTemporal(embed_dim)
        # self.decoder_hw = nn.Linear(40, 64)
        # self.decoder_hw = nn.Sequential(nn.Linear(144, 64),
        #                                 nn.ReLU)
        # self.decoder_time1 = nn.Linear(x_10_trj_len+x_10_len, cfg.process_frame_nums-cfg.fps)
        self.decoder_time1 = nn.Linear(6, cfg.process_frame_nums - cfg.fps)
        # self.decoder_time2 = nn.Linear(9, 30)
        # self.head1 = nn.Linear(9, 1)
        # self.head1 = nn.Linear(10, 1)
        # self.norm_fus = norm_layer(9)
        self.norm_pre = norm_layer(cfg.MVIT.EMBED_DIM)
        self.norm_pre_trj = norm_layer(cfg.MVIT.EMBED_DIM)
        self.norm_timeemb = norm_layer(cfg.MVIT.EMBED_DIM)
        self.norm_timeemb_trj = norm_layer(cfg.MVIT.EMBED_DIM)
        self.norm = norm_layer(dim_out)
        self.norm_trj = norm_layer(cfg.MVIT.EMBED_DIM)
        # self.head2 = nn.Linear(cfg.MVIT.EMBED_DIM * 4, 1)
        self.class_embedding = nn.Parameter(torch.randn(cfg.MVIT.EMBED_DIM))
        self.class_embedding_trj = nn.Parameter(torch.randn(cfg.MVIT.EMBED_DIM))
        self.ln_post = LayerNorm(embed_dim)
        self.ln_post1 = LayerNorm(embed_dim)
        self.ln_post_trj = LayerNorm(embed_dim)
        self.upsampling_head = nn.Linear(x_10_trj_len, x_10_len)
        self.downsampling_head = nn.Sequential(nn.Linear(x_10_len, x_10_trj_len),
                                               nn.ReLU())
        # self.norm1 = LayerNorm(cfg.MVIT.EMBED_DIM)
        # self.norm2 = LayerNorm(cfg.MVIT.EMBED_DIM * 2)
        # self.norm3 = LayerNorm(cfg.MVIT.EMBED_DIM * 4)
        self.dropout = nn.Dropout(0.3)
        self.droppath = DropPath(0.3)
        self.norm_fina = norm_layer(144)
        self.norm_mvit = norm_layer(embed_dim)
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.time_embed, std=0.02)
            trunc_normal_(self.pos_embed_trj, std=0.02)
            trunc_normal_(self.class_embedding, std=.02)
            trunc_normal_(self.class_embedding_trj, std=.02)
            # trunc_normal_(self.head_token, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # self.att_upsample = att_upsample(x_len= x_len, x_10_len=x_10_len, x_15_len=x_15_len)
        # self.att_downsample = att_downsample()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward(self, x):
        histry_traj = int(self.fps / self.fps_stride)
        histry_traj_10 = int((histry_traj+1) / self.fps10_stride)
        if isinstance(x, tuple):
            headmaps, salmaps = x

            B = headmaps.shape[0]

            # B, T_head, C, _, _ = headmaps.shape
            if self.use_sal:
                salmaps = salmaps.unsqueeze(dim=1)  ##(16,60,144,256)
                headmaps = headmaps.unsqueeze(dim=1)

                headmaps_feature_10, bcthw_trj = self.patch_embed(headmaps)
                # headmaps_feature = self.relu(headmaps_feature)
                # headmaps_feature_10, bcthw_trj = self.patch_embed_10(headmaps_feature)
                # headmaps_feature_10 = self.relu(headmaps_feature_10)
                headmaps_feature_10 = headmaps_feature_10.flatten(3).transpose(1, 2)
                # headmaps_feature_10 = rearrange(headmaps_feature_10, 'b t c l -> b c l t')
                # headmaps_feature_10 = self.upsampling_head(headmaps_feature_10)
                # headmaps_feature_10 = rearrange(headmaps_feature_10, 'b c l t -> b t c l')
                # headmaps_feature_10 = headmaps_feature_10.flatten(3).transpose(1, 2)

                salfeature_10, bchw_sal = self.patch_embed_sal(salmaps)

                salfeature_10 = salfeature_10.flatten(3).transpose(1, 2)


                x_10 = salfeature_10
                x_10_trj = headmaps_feature_10
                B_10, T_10, C_10 = x_10.shape[0], x_10.shape[1], x_10.shape[2]
                B_10_trj, T_10_trj, C_10_trj = x_10_trj.shape[0], x_10_trj.shape[1], x_10_trj.shape[2]
            else:
                headmaps, bchw = self.patch_embed(headmaps)  # ->   (8,5,40,928)
                # T = T_head
            ## Pos Embeddings
            if self.use_abs_pos:
                if self.use_sal:
                    x_10 = rearrange(x_10, 'b t c l -> (b t) l c')
                    x_10 = x_10 + self.pos_embed
                    # x_10 = self.norm_pre(x_10)
                    # x_10 = x_10.permute(1, 0, 2)

                    x_10_trj = rearrange(x_10_trj, 'b t c l -> (b t) l c')
                    x_10_trj = x_10_trj + self.pos_embed_trj
                    # x_10_trj = self.norm_pre_trj(x_10_trj)
                    # x_10_trj = x_10_trj.permute(1, 0, 2)

                else:
                    x = headmaps + self.pos_embed
            # H_10, W_10 = bchw[-2], bchw[-1]
            H_10, W_10 = bchw_sal[-2], bchw_sal[-1]
            H_10_trj, W_10_trj = bcthw_trj[-2], bcthw_trj[-1]
            thw_10 = [T_10, H_10, W_10]  # b   t   n   c
            thw_10_trj = [T_10_trj, H_10_trj, W_10_trj]
            ##  Time Embeddings
            x_10 = rearrange(x_10, '(b t) l c -> (b l) t c', b=B)
            x_10 = x_10 + self.time_embed
            x_10 = rearrange(x_10, '(b l) t c -> (b l) c t', b=B)
            x_10 = self.downsampling_head(x_10)
            x_10 = rearrange(x_10, '(b l) c t -> b (t l) c', b=B)
            x_10 = self.norm_timeemb(x_10)

            x_10_trj = rearrange(x_10_trj, '(b t) l c -> (b l) t c', b=B)
            x_10_trj = x_10_trj + self.time_embed_trj
            x_10_trj = rearrange(x_10_trj, '(b l) t c -> b (t l) c', b=B)
            x_10_trj = self.norm_timeemb_trj(x_10_trj)

        ## 先mvit 空间特征
        # x_10 = rearrange(x_10, 'n (b t) d -> b (t n) d', b=B)
        # x_10_trj = rearrange(x_10_trj, 'n (b t) d -> b (t n) d', b=B)

        for blk in self.blocks:
            # blk_trj = self.fuse_blocks[index]
            # x_10_trj, thw_10_trj = blk(x_10_trj, thw_10_trj)  # 2   11136    40
            x_10, thw_10_trj, x_10_trj = blk(x_10, thw_10_trj, x_10_trj)
            # index += 1
        thw_10_fus = [2*thw_10_trj[0], thw_10_trj[1], thw_10_trj[2]]
        x_fus = torch.cat((x_10, x_10_trj), dim=1)
        x_fus = self.ln_post1(x_fus)
        for index in range(4):
            blk_trj = self.fuse_blocks[index]
            x_fus, thw_10_fus = blk_trj(x_fus, thw_10_fus)
        x_10 = self.ln_post(x_fus)
        x_10 = rearrange(x_10, 'b (t n) c -> b n c t', t=thw_10_fus[0])
        x = x_10
        x_out = self.dropout(self.decoder_time1(x))

        ##再加mvit 时间过长
        # x_out = rearrange(x_out, 'b t (h w) c -> b (h w) t c', h=8, w=8)
        x_out = rearrange(x_out, 'b n c t -> b n t c')
        x_out = self.decoder_emb2(self.decoder_emb1(x_out)).squeeze(dim=-1)
        x_out = self.relu(x_out)
        x_out = rearrange(x_out, 'b l t -> b t l')
        # x_out = self.decoder_hw(x_out)
        x_out = self.norm_fina(x_out)

        ## decoder用卷积
        x_out = self.output_softmax(x_out)
        return x_out


def _prepare_mvit_configs(cfg):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    depth = cfg.MVIT.DEPTH
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    pool_t = [[] for i in range(depth)]
    for i in range(len(cfg.MVIT.DIM_MUL)):
        dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
    for i in range(len(cfg.MVIT.HEAD_MUL)):
        head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]
    for i in range(len(cfg.MVIT.POOL_T_STRIDE)):
        pool_t[cfg.MVIT.POOL_T_STRIDE[i][0]] = cfg.MVIT.POOL_T_STRIDE[i][1:]
    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    fuse_stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]
    fuse_stride_kv = [[] for i in range(depth)]

    for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
        stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][1:]
        fuse_stride_q[cfg.MVIT.SAL_POOL_Q_STRIDE[i][0]] = cfg.MVIT.SAL_POOL_Q_STRIDE[i][1:]
        pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
        _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
        cfg.MVIT.POOL_KV_STRIDE = []
        for i in range(cfg.MVIT.DEPTH):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

    if cfg.MVIT.SAL_POOL_KV_STRIDE_ADAPTIVE is not None:
        _sal_stride_kv = cfg.MVIT.SAL_POOL_KV_STRIDE_ADAPTIVE
        cfg.MVIT.SAL_POOL_KV_STRIDE = []
        for i in range(cfg.MVIT.DEPTH):
            if len(fuse_stride_q[i]) > 0:
                _sal_stride_kv = [
                    max(_sal_stride_kv[d] // fuse_stride_q[i][d], 1)
                    for d in range(len(_sal_stride_kv))
                ]
            cfg.MVIT.SAL_POOL_KV_STRIDE.append([i] + _sal_stride_kv)

    for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
        stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[i][1:]
        fuse_stride_kv[cfg.MVIT.SAL_POOL_KV_STRIDE[i][0]] = cfg.MVIT.SAL_POOL_KV_STRIDE[i][1:]
        pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv, pool_t, fuse_stride_q, fuse_stride_kv
    # return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv, pool_t
