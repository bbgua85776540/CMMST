#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


import numpy
import torch
import torch.nn as nn
from mvit.models.common import DropPath, Mlp
from torch.nn.init import trunc_normal_
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
from einops import rearrange



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

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
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.down_proj(x)
        x = self.activation(self.conv(x))
        x = self.up_proj(x)

        return x


def attention_pool(tensor, pool, hw_shape, pool_t=None, has_cls_embed=True, norm=None):
    if pool is None:
        if pool_t is not None:
            if tensor.ndim == 3: tensor = tensor.unsqueeze(1)
            B, N, L, C = tensor.shape
            T, H, W = hw_shape
            tensor = rearrange(tensor, 'b n (t h w) c -> (b n c) t h w', t=T, h=H, w=W, b=B,
                               n=N, c=tensor.shape[-1])
            tensor = pool_t(tensor)
            T = tensor.shape[1]
            tensor = rearrange(tensor, '(b n c) t h w-> (b n t) c h w', t=tensor.shape[1], h=tensor.shape[2],
                               w=tensor.shape[3], b=B, n=N, c=C)
            hw_shape = [T, tensor.shape[2], tensor.shape[3]]
            L_pooled = tensor.shape[2] * tensor.shape[3] * T
            tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)  # tensor= torch.Size([2, 1, 23040, 96])
            if norm is not None:
                tensor = norm(tensor)
            if tensor.ndim == 3: tensor = tensor.squeeze(1)
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape  # (2,1,23040,96)
    T, H, W = hw_shape
    tensor = rearrange(tensor, 'b n (t h w) c-> (b n t) c h w', t=T, h=H, w=W, b=B, n=N,
                       c=C).contiguous()  # (2,1,5568,6)

    tensor = pool(tensor)

    if pool_t is not None:
        tensor = rearrange(tensor, '(b n t) c h w-> (b n c) t h w', t=T, h=tensor.shape[2], w=tensor.shape[3], b=B, n=N,
                           c=tensor.shape[1])
        tensor = pool_t(tensor)
        T = tensor.shape[1]
        tensor = rearrange(tensor, '(b n c) t h w-> (b n t) c h w', t=tensor.shape[1], h=tensor.shape[2],
                           w=tensor.shape[3], b=B, n=N,
                           c=C)
    hw_shape = [T, tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * T
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)  # tensor= torch.Size([2, 1, 23040, 96])
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


def cal_rel_pos_spatial(
        attn,
        q,
        has_cls_embed,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)  # b y  h    w   c == h  k   c
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)  # 2,10,36, 64, 96 ==36, 9 , 96 ; 2,10,56,56,96==56,14,96
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            input_size,
            num_heads=8,
            qkv_bias=False,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            norm_layer=nn.LayerNorm,
            has_cls_embed=True,
            mode="conv",
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_zero_init=False,
            residual_pooling=True,
            pool_t=None
    ):
        super().__init__()
        self.pool_first = pool_first

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.pool_t = (
                nn.Conv2d(
                    pool_t[0],
                    pool_t[0] // pool_t[1],
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 0),
                    # groups=pool_t[0]//pool_t[1],
                    bias=False,
                )
                if pool_t[1] > 1
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    # forward MultiScaleAttention
    def forward(self, x, hw_shape):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # (2,1,23040,96)

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
            pool_t=self.pool_t,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
            pool_t=self.pool_t,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
            pool_t=self.pool_t,
        )

        if self.pool_first:
            q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
            k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
            v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x, q_shape



class MultiScaleCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            input_size,
            num_heads=8,
            qkv_bias=False,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            norm_layer=nn.LayerNorm,
            has_cls_embed=True,
            mode="conv",
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_zero_init=False,
            residual_pooling=True,
            pool_t=None
    ):
        super().__init__()
        self.pool_first = pool_first

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.pool_t = (
                nn.Conv2d(
                    pool_t[0],
                    pool_t[0] // pool_t[1],
                    (3, 3),
                    stride=(1, 1),
                    padding=(0, 0),
                    # groups=pool_t[0]//pool_t[1],
                    bias=False,
                )
                if pool_t[1] > 1
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    # forward MultiScaleAttention
    def forward(self, x, hw_shape, y):
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            qkv_y = (
                self.qkv(y).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q_y, k_y, v_y = qkv[0], qkv_y[1], qkv_y[2]  # (2,1,23040,96)
            q, k, v = qkv_y[0], qkv[1], qkv[2]  # (2,1,23040,96)

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
            pool_t=self.pool_t,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
            pool_t=self.pool_t,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
            pool_t=self.pool_t,
        )
        q_y, q_shape_y = attention_pool(
            q_y,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
            pool_t=self.pool_t,
        )
        k_y, k_shape_y = attention_pool(
            k_y,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
            pool_t=self.pool_t,
        )
        v_y, v_shape_y = attention_pool(
            v_y,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
            pool_t=self.pool_t,
        )

        if self.pool_first:
            q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
            k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
            v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # if self.rel_pos_spatial:
        #     attn = cal_rel_pos_spatial(
        #         attn,
        #         q,
        #         self.has_cls_embed,
        #         q_shape,
        #         k_shape,
        #         self.rel_pos_h,
        #         self.rel_pos_w,
        #     )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        attn_y = (q_y * self.scale) @ k_y.transpose(-2, -1)
        attn_y = attn_y.softmax(dim=-1)
        y = attn_y @ v_y

        if self.residual_pooling:
            if self.has_cls_embed:
                y[:, :, 1:, :] += q_y[:, :, 1:, :]
            else:
                y = y + q_y

        y = y.transpose(1, 2).reshape(B, -1, self.dim_out)
        y = self.proj(y)
        return x, q_shape, y


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            input_size,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            mode="conv",
            has_cls_embed=True,
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_zero_init=False,
            residual_pooling=True,
            dim_mul_in_att=False,
            pool_t=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att

        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleCrossAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            pool_t=pool_t
        )
        self.pool_t = self.pool_t = (
            nn.Conv2d(
                pool_t[0],
                pool_t[0] // pool_t[1],
                (3, 3),
                stride=(1, 1),
                padding=(0, 0),
                # groups=pool_t[0]//pool_t[1],
                bias=False,
            )
            if pool_t[1] > 1
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None

    # foward MultiScaleBlock
    def forward(self, x, hw_shape, y):

        x_norm = self.norm1(x)
        y_norm = self.norm1(y)
        x_block, hw_shape_new, y_block = self.attn(x_norm, hw_shape, y)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
            y = self.proj(y_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed, pool_t=self.pool_t
        )
        y_res, _ = attention_pool(
            y, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed, pool_t=self.pool_t
        )
        if len(x_res.shape) > 3:
            x_res = rearrange(x_res, 'b t n c -> b (t n) c')
            y_res = rearrange(y_res, 'b t n c -> b (t n) c')
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)

        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        y = y_res + self.drop_path(y_block)
        y_norm = self.norm2(y)
        y_mlp = self.mlp(y_norm)

        if not self.dim_mul_in_att and self.dim != self.dim_out:
            y = self.proj(y_norm)
        y = y + self.drop_path(y_mlp)

        return x, hw_shape_new, y


class MultiScaleFuseBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            input_size,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            kernel_q=(1, 1),
            kernel_kv=(1, 1),
            stride_q=(1, 1),
            stride_kv=(1, 1),
            mode="conv",
            has_cls_embed=True,
            pool_first=False,
            rel_pos_spatial=False,
            rel_pos_zero_init=False,
            residual_pooling=True,
            dim_mul_in_att=False,
            pool_t=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.dim_mul_in_att = dim_mul_in_att

        att_dim = dim_out if dim_mul_in_att else dim
        self.attn = MultiScaleAttention(
            dim,
            att_dim,
            num_heads=num_heads,
            input_size=input_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
            pool_t=pool_t
        )
        self.pool_t = self.pool_t = (
            nn.Conv2d(
                pool_t[0],
                pool_t[0] // pool_t[1],
                (3, 3),
                stride=(1, 1),
                padding=(0, 0),
                # groups=pool_t[0]//pool_t[1],
                bias=False,
            )
            if pool_t[1] > 1
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(att_dim)
        mlp_hidden_dim = int(att_dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed

        mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if len(stride_q) > 0 and numpy.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None

    # foward MultiScaleBlock
    def forward(self, x, hw_shape):

        x_norm = self.norm1(x)
        x_block, hw_shape_new = self.attn(x_norm, hw_shape)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed, pool_t=self.pool_t
        )
        if len(x_res.shape) > 3:
            x_res = rearrange(x_res, 'b t n c -> b (t n) c')
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if not self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)

        # y_norm = self.norm1(y)
        # y_block, hw_shape_new_y = self.attn(y_norm, hw_shape)
        # if self.dim_mul_in_att and self.dim != self.dim_out:
        #     y = self.proj(y_norm)
        # y_res, _ = attention_pool(
        #     y, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed, pool_t=self.pool_t
        # )
        # if len(y_res.shape) > 3:
        #     y_res = rearrange(y_res, 'b t n c -> b (t n) c')
        # y = y_res + self.drop_path(y_block)
        # y_norm = self.norm2(y)
        # y_mlp = self.mlp(y_norm)
        # if not self.dim_mul_in_att and self.dim != self.dim_out:
        #     y = self.proj(y_norm)
        # y = y + self.drop_path(y_mlp)

        return x, hw_shape_new
