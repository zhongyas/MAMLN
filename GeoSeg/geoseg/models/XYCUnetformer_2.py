import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import math


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # print(coords_h, coords_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print(coords_flatten)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # print(relative_coords[0,7,:])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # print(relative_coords[:, :, 0], relative_coords[:, :, 1])
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(B_,N,C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias.unsqueeze(0))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)#nW, window_size*window_size
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)#nW, window_size*window_size, window_size*window_size
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))#nW, window_size*window_size, window_size*window_size

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)#(4,4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)#B Wh*Ww C
            x = self.norm(x)#B Wh*Ww C
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)#B C Wh Ww

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]#56,56

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))#1,128,56,56
            trunc_normal_(self.absolute_pos_embed, std=.02)#作用是将输入的Tensor用均值为0.0，标准差为std的正态分布进行填充

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth、 stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]#作用是在0和drop_path_rate之间均匀取sum(depths)个数

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        # print('input size', x.size())#torch.Size([6, 3, 512, 512])
        x = self.patch_embed(x)
        # print('patch_embed', x.size())#torch.Size([6, 128, 128, 128])

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        # print('x size', x.size())#([6, 16384, 128])
        x = self.pos_drop(x)
        # print('pos_drop', x.size())#pos_drop torch.Size([6, 16384, 128])

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # print('layer{} input size {}'.format(i, x.size()))
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # print('x size', x.size(),Wh,Ww,x_out.size())
            '''
            layer0 input size torch.Size([6, 16384, 128])
            x size torch.Size([6, 4096, 256]) 64 64
            layer1 input size torch.Size([6, 4096, 256])
            x size torch.Size([6, 1024, 512]) 32 32
            layer2 input size torch.Size([6, 1024, 512])
            x size torch.Size([6, 256, 1024]) 16 16
            layer3 input size torch.Size([6, 256, 1024])
            x size torch.Size([6, 256, 1024]) 16 16
            '''
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                
                # print('layer{} out size {}'.format(i, out.size()))

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class SwinTransformer2(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]#56,56

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))#1,128,56,56
            trunc_normal_(self.absolute_pos_embed, std=.02)#作用是将输入的Tensor用均值为0.0，标准差为std的正态分布进行填充

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth、 stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]#作用是在0和drop_path_rate之间均匀取sum(depths)个数

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,x24,x23,x22,x21):
        """Forward function."""
        # print('input size', x.size())#torch.Size([6, 3, 512, 512])
        x = self.patch_embed(x)
        # print('patch_embed', x.size())#torch.Size([6, 128, 128, 128])
        # print('x24',x24.size())
        # print('x23',x23.size())
        # print('x22',x22.size())
        # print('x21',x21.size())

        x24 = x24.flatten(2).transpose(1, 2)
        x23 = x23.flatten(2).transpose(1, 2)
        x22 = x22.flatten(2).transpose(1, 2)
        x21 = x21.flatten(2).transpose(1, 2)

        # print('x24!',x24.size())
        # print('x23!',x23.size())
        # print('x22!',x22.size())
        # print('x21!',x21.size())
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        # print('x size', x.size())#([6, 16384, 128])
        x = self.pos_drop(x)
        # print('pos_drop', x.size())#pos_drop torch.Size([6, 16384, 128])

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            # print('layer{} input size {}'.format(i, x.size()))
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # print('x size', x.size(),Wh,Ww,x_out.size())
            '''
            layer0 input size torch.Size([6, 16384, 128])
            x size torch.Size([6, 4096, 256]) 64 64
            layer1 input size torch.Size([6, 4096, 256])
            x size torch.Size([6, 1024, 512]) 32 32
            layer2 input size torch.Size([6, 1024, 512])
            x size torch.Size([6, 256, 1024]) 16 16
            layer3 input size torch.Size([6, 256, 1024])
            x size torch.Size([6, 256, 1024]) 16 16
            '''
            
            if i in self.out_indices:
                if i==0:
                    x_out=x_out+x24
                elif i==1:
                    x_out=x_out+x23
                elif i==2:
                    x_out=x_out+x22
                else:
                    x_out=x_out+x21
                
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                
                # print('layer{} out size {}'.format(i, out.size()))

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer2, self).train(mode)
        self._freeze_stages()



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp_decoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size


        self.prechannel = ConvBN(dim, dim, kernel_size=1)
        self.localx1= ConvBN(86, 86, kernel_size=3,dilation=1)
        self.localx2 = ConvBN(86, 86, kernel_size=3, dilation=2)
        self.localx3 = ConvBN(84, 84, kernel_size=3, dilation=3)



        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        #local = self.local2(x) + self.local1(x)
        local=self.prechannel(x)#size: B,C,H,W
        # 将输入张量的通道维度平均分为 3 份
        local1, local2, local3 = torch.chunk(local, 3, dim=1)
        # print("local1",local1.size())
        # print("local",local2.size())
        # print("local",local3.size())
        localx1=self.localx1(local1)
        localx2=self.localx2(local2)
        localx3=self.localx3(local3)
        # print("local",local.size())
        # print("localx1",localx1.size())
        # print("localx1",localx2.size())
        # print("localx1",localx3.size())

        # 分别对 3 个通道进行卷积操作
        local = torch.cat((localx1,localx2,localx3), dim=1)
        # print("!!1",local.size())
        # local=torch.cat((localx3(localx3),local),dim=1)
        
            


        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        # print('self.norm1(x).size(),self.norm2(x).size()=',self.norm1(x).size(),self.norm2(x).size())
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)#fuse_weights[0]代表x的权重，fuse_weights[1]代表res的权重，两者相加为1，fuse_weights[0]是可学习的，fuse_weights[1]是1-fuse_weights[0]
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WF2(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF2, self).__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)#可学习的权重
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):#x和res的尺寸一致
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)#fuse_weights[0]代表x的权重，fuse_weights[1]代表res的权重，两者相加为1，fuse_weights[0]是可学习的，fuse_weights[1]是1-fuse_weights[0]
        # print(fuse_weights.size())
        # print(x.size(),res.size())
        # print(self.pre_conv(res).size())#
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

class CNNStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, act_layer=nn.GELU,norm_layer=nn.BatchNorm2d, bias=False):
        super(CNNStem, self).__init__()
        self.convbn1=ConvBNReLU(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=2, norm_layer=norm_layer, bias=bias)
        self.convbn2=ConvBNReLU(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1, norm_layer=norm_layer, bias=bias)
        self.convbn3=ConvBNReLU(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1, norm_layer=norm_layer, bias=bias)
        self.convbn4=ConvBNReLU(out_channels, out_channels, kernel_size=1, dilation=dilation, stride=1, norm_layer=norm_layer, bias=bias)
    def forward(self, x):
        x=self.convbn1(x)
        x=self.convbn2(x)
        x=self.convbn3(x)
        x=self.convbn4(x)
        return x

class PSPModule(nn.Module):
    """
     *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class PointMatcher(nn.Module):
    """
        Simple Point Matcher
    """
    def __init__(self, dim, kernel_size=3):
        super(PointMatcher, self).__init__()
        self.match_conv = nn.Conv2d(dim*2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_high, x_low = x
        x_low = F.upsample(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        certainty = self.match_conv(torch.cat([x_high, x_low], dim=1))
        return self.sigmoid(certainty)

class PointFlowModuleWithMaxAvgpool(nn.Module):
    def __init__(self, in_planes,  dim=64, maxpool_size=8, avgpool_size=8, matcher_kernel_size=3,
                  edge_points=64):
        super(PointFlowModuleWithMaxAvgpool, self).__init__()
        self.dim = dim
        self.point_matcher = PointMatcher(dim, matcher_kernel_size)#PointMatcher
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool_size = maxpool_size
        self.avgpool_size = avgpool_size
        self.edge_points = edge_points
        self.max_pool = nn.AdaptiveMaxPool2d((maxpool_size, maxpool_size), return_indices=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((avgpool_size, avgpool_size))
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            # Norm2d(in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x_high, x_low = x
        stride_ratio = x_low.shape[2] / x_high.shape[2]
        x_high_embed = self.down_h(x_high)
        x_low_embed = self.down_l(x_low)
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        certainty_map = self.point_matcher([x_high_embed, x_low_embed])
        avgpool_grid = self.avg_pool(certainty_map)
        _, _, map_h, map_w = certainty_map.size()
        avgpool_grid = F.interpolate(avgpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)

        # edge part
        x_high_edge = x_high - x_high * avgpool_grid
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred, num_points=self.edge_points)
        sample_x = point_indices % W_h * stride_ratio
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)
        low_edge_feat = point_sample(x_low, point_coords)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        fusion_edge_feat = high_edge_feat + low_edge_feat

        # residual part
        maxpool_grid, maxpool_indices = self.max_pool(certainty_map)
        maxpool_indices = maxpool_indices.expand(-1, C, -1, -1)
        maxpool_grid = F.interpolate(maxpool_grid, size=(map_h, map_w), mode="bilinear", align_corners=True)
        x_indices = maxpool_indices % W_h * stride_ratio
        y_indices = maxpool_indices // W_h * stride_ratio
        low_indices = x_indices + y_indices * W
        low_indices = low_indices.long()
        x_high = x_high + maxpool_grid*x_high
        flattened_high = x_high.flatten(start_dim=2)
        high_features = flattened_high.gather(dim=2, index=maxpool_indices.flatten(start_dim=2)).view_as(maxpool_indices)
        flattened_low = x_low.flatten(start_dim=2)
        low_features = flattened_low.gather(dim=2, index=low_indices.flatten(start_dim=2)).view_as(low_indices)
        feat_n, feat_c, feat_h, feat_w = high_features.shape
        high_features = high_features.view(feat_n, -1, feat_h*feat_w)
        low_features = low_features.view(feat_n, -1, feat_h*feat_w)
        affinity = torch.bmm(high_features.transpose(2, 1), low_features).transpose(2, 1)
        affinity = self.softmax(affinity)  # b, n, n
        high_features = torch.bmm(affinity, high_features.transpose(2, 1)).transpose(2, 1)
        fusion_feature = high_features + low_features
        mp_b, mp_c, mp_h, mp_w = low_indices.shape
        low_indices = low_indices.view(mp_b, mp_c, -1)

        final_features = x_low.reshape(N, C, H*W).scatter(2, low_edge_indices, fusion_edge_feat)
        final_features = final_features.scatter(2, low_indices, fusion_feature).view(N, C, H, W)

        return final_features, edge_pred



class Decoder1(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),#(128, 256, 512, 1024)
                 decode_channels=64,#256
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder1, self).__init__()
        # print('encoder_channels',encoder_channels,'decode_channels',decode_channels)#encoder_channels [128, 256, 512, 1024] decode_channels 256
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.prea_conv=ConvBN(encoder_channels[-2], decode_channels, kernel_size=1)
        self.pre_conv1 = ConvBN(in_channels=decode_channels,out_channels=decode_channels*2, kernel_size=1)
        self.prea_conv1 = ConvBN(in_channels=256,out_channels=128, kernel_size=1)
        self.pre_conv2 = ConvBN(in_channels=128,out_channels=256, kernel_size=1)
            
        self.b4 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        

        #self.ppm = PSPModule(encoder_channels[-1], out_features=decode_channels)
        self.ppm = PSPModule(encoder_channels[-1], out_features=encoder_channels[-1])#(1024, out_features=1024)
        self.cnnstem4=CNNStem(in_channels=encoder_channels[-2], out_channels=encoder_channels[-1])#(512, 1024)
        self.cnnstem3=CNNStem(in_channels=encoder_channels[-3], out_channels=encoder_channels[-2])#(256, 512)
        self.cnnstem2=CNNStem(in_channels=encoder_channels[-4], out_channels=encoder_channels[-3])#(128, 256)
        self.cnnstem1=CNNStem(in_channels=encoder_channels[-4], out_channels=encoder_channels[-4])#(128, 128)
        self.cnnstem1a=ConvBNReLU(in_channels=encoder_channels[-4], out_channels=encoder_channels[-4], kernel_size=1)#(128, 128)

        self.pp4=WF2(in_channels=encoder_channels[-1],decode_channels=encoder_channels[-1] )#(1024, 1024)
        self.pp3=WF2(in_channels=encoder_channels[-2],decode_channels=encoder_channels[-2] )#(512, 512)
        self.pp2=WF2(in_channels=encoder_channels[-3],decode_channels=encoder_channels[-3] )#(256, 256)
        self.pp1=WF2(in_channels=encoder_channels[-4],decode_channels=encoder_channels[-4] )#(128, 128)
        self.pfm3=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-2], dim=decode_channels, maxpool_size=8,
                                                  avgpool_size=8, edge_points=32)
        self.pfm2=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-3], dim=decode_channels, maxpool_size=8,
                                                    avgpool_size=8, edge_points=32)
        self.pfm1=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-4], dim=decode_channels, maxpool_size=8,
                                                    avgpool_size=8, edge_points=32)

        self.b3 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):

        # print('res1', res1.size())
        # print('res2', res2.size())  
        # print('res3', res3.size())
        # print('res4', res4.size())
        x11 = self.b4(self.pre_conv(res4))
        # print('b4', x.size())
        x12 = self.p3(x11, res3)
        # print('p3', x.size())
        x12 = self.b3(x12)
        # print('b3', x.size())

        x13 = self.p2(x12, res2)
        # print('p2', x.size())
        x13 = self.b2(x13)
        # print('b2', x.size())

        x14 = self.p1(x13, res1)
        # print('p1', x.size())

        x = self.segmentation_head(x14)
        # print('segmentation_head', x.size())
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # print('interpolate', x.size())
        # print('x11,x12,x13,x14',x11.size(),x12.size(),x13.size(),x14.size())
        return x11,x12,x13,x14

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder2(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),#(128, 256, 512, 1024)
                 decode_channels=64,#256
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder2, self).__init__()
        # print('encoder_channels',encoder_channels,'decode_channels',decode_channels)#encoder_channels [128, 256, 512, 1024] decode_channels 256
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.prea_conv=ConvBN(encoder_channels[-2], decode_channels, kernel_size=1)
        self.pre_conv1 = ConvBN(in_channels=decode_channels,out_channels=decode_channels*2, kernel_size=1)
        self.prea_conv1 = ConvBN(in_channels=256,out_channels=128, kernel_size=1)
        self.pre_conv2 = ConvBN(in_channels=128,out_channels=256, kernel_size=1)
            
        self.b4 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        

        #self.ppm = PSPModule(encoder_channels[-1], out_features=decode_channels)
        self.ppm = PSPModule(encoder_channels[-1], out_features=encoder_channels[-1])#(1024, out_features=1024)
        self.cnnstem4=CNNStem(in_channels=encoder_channels[-2], out_channels=encoder_channels[-1])#(512, 1024)
        self.cnnstem3=CNNStem(in_channels=encoder_channels[-3], out_channels=encoder_channels[-2])#(256, 512)
        self.cnnstem2=CNNStem(in_channels=encoder_channels[-4], out_channels=encoder_channels[-3])#(128, 256)
        self.cnnstem1=CNNStem(in_channels=encoder_channels[-4], out_channels=encoder_channels[-4])#(128, 128)
        self.cnnstem1a=ConvBNReLU(in_channels=encoder_channels[-4], out_channels=encoder_channels[-4], kernel_size=1)#(128, 128)

        self.pp4=WF2(in_channels=encoder_channels[-1],decode_channels=encoder_channels[-1] )#(1024, 1024)
        self.pp3=WF2(in_channels=encoder_channels[-2],decode_channels=encoder_channels[-2] )#(512, 512)
        self.pp2=WF2(in_channels=encoder_channels[-3],decode_channels=encoder_channels[-3] )#(256, 256)
        self.pp1=WF2(in_channels=encoder_channels[-4],decode_channels=encoder_channels[-4] )#(128, 128)
        self.pfm3=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-2], dim=decode_channels, maxpool_size=8,
                                                  avgpool_size=8, edge_points=32)
        self.pfm2=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-3], dim=decode_channels, maxpool_size=8,
                                                    avgpool_size=8, edge_points=32)
        self.pfm1=PointFlowModuleWithMaxAvgpool(in_planes=encoder_channels[-4], dim=decode_channels, maxpool_size=8,
                                                    avgpool_size=8, edge_points=32)

        self.b3 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):

        # print('res1', res1.size())
        # print('res2', res2.size())  
        # print('res3', res3.size())
        # print('res4', res4.size())
        x11 = self.b4(self.pre_conv(res4))
        # print('b4', x.size())
        x11 = self.p3(x11, res3)
        # print('p3', x.size())
        x12 = self.b3(x11)
        # print('b3', x.size())

        x12 = self.p2(x12, res2)
        # print('p2', x.size())
        x13 = self.b2(x12)
        # print('b2', x.size())

        x14 = self.p1(x13, res1)
        # print('p1', x.size())

        x = self.segmentation_head(x14)
        # print('segmentation_head', x.size())
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # print('interpolate', x.size())

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        kernel_size = [1,3,3,1]
        dilation = [1,3,6,1]
        paddings= [0, 3,6,0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_size)):
            conv=torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size[aspp_idx], stride=1, padding=paddings[aspp_idx], dilation=dilation[aspp_idx], bias=True)
            self.aspp.append(conv)
            # conv.bias.data.fill_(0)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num=len(kernel_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                m.bias.data.fill_(0)
    def forward(self, x):
        avg_x=self.gap(x)
        out=[]
        for aspp_idx in range(self.aspp_num):
            inp=avg_x if (aspp_idx==self.aspp_num -1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1]=out[-1].expand_as(out[-2])
        out=torch.cat(out,dim=1)
        return out



class XYCUNetFormer_1(nn.Module):

    def __init__(self,
                 decode_channels=256,
                 decode_channelsxyc=1024,
                 dropout=0.2,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 freeze_stages=-1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        #swin b
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=freeze_stages)
        self.backbone2 = SwinTransformer2(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=freeze_stages)
        encoder_channels = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]#128,256,512,1024
        self.decoder1 = Decoder1(encoder_channels, decode_channels, dropout, window_size, num_classes)
        self.decoder2 = Decoder2(encoder_channels, decode_channels, dropout, window_size, num_classes)
        self.aspp=ASPP(in_channels=decode_channels, out_channels=decode_channels)
        self.pre_conv1 = ConvBN(in_channels=decode_channelsxyc,out_channels=encoder_channels[0], kernel_size=1)
        self.pre_conv2 = ConvBN(in_channels=decode_channelsxyc,out_channels=encoder_channels[1], kernel_size=1)
        self.pre_conv3 = ConvBN(in_channels=decode_channelsxyc,out_channels=encoder_channels[2], kernel_size=1)
        self.pre_conv4 = ConvBN(in_channels=decode_channelsxyc,out_channels=encoder_channels[3],kernel_size=1)


    def forward(self, x):
        h, w = x.size()[-2:]
        # print("1x.shape=",x.shape)#x.shape= torch.Size([8, 3, 512, 512])
        res1, res2, res3, res4 = self.backbone(x)
        # print("res1.shape=",res1.shape)
        # print("res2.shape=",res2.shape)
        # print("res3.shape=",res3.shape)
        # print("res4.shape=",res4.shape)
        x11,x12,x13,x14 = self.decoder1(res1, res2, res3, res4, h, w)
        # print("x11.shape=",x11.shape)
        # print("x12.shape=",x12.shape)
        # print("x13.shape=",x13.shape)
        # print("x14.shape=",x14.shape)
        # x11.shape= torch.Size([4, 256, 64, 64])
        # x12.shape= torch.Size([4, 256, 128, 128])
        # x13.shape= torch.Size([4, 256, 128, 128])
        # x14.shape= torch.Size([4, 256, 256, 256])
        x21=self.aspp(x11)
        # print("x21.shape=",x21.shape)#x21.shape= torch.Size([4, 1024, 64, 64])

        x21=self.pre_conv4(x21)

        x22=self.aspp(x12)
        # print("x22.shape=",x22.shape)#
        x22=self.pre_conv3(x22)

        x23=self.aspp(x13)
        # print("x23.shape=",x23.shape)#
        x23=self.pre_conv2(x23)

        x24=self.aspp(x14)
        # print("x24.shape=",x24.shape)#
        x24=self.pre_conv1(x24)
        # print("x21.shape=",x21.shape)
        # print("x22.shape=",x22.shape)
        # print("x23.shape=",x23.shape)
        # print("x24.shape=",x24.shape)

        #############################
        # x24=x24+res1
        # x23=x23+res2
        # x22=x22+res3
        # x21=x21+res4
        #############################

        res1, res2, res3, res4 = self.backbone2(x,x24,x23,x22,x21)

        x=self.decoder2(res1,res2,res3,res4,h,w)
        # print("@@@x.shape=",x.shape)
        return x


def xyc_unetformer(pretrained=True, num_classes=6, freeze_stages=-1, decoder_channels=256,
                  weight_path='pretrain_weights/stseg_base.pth'):
    model = XYCUNetFormer_1(num_classes=num_classes,
                         freeze_stages=freeze_stages,
                         embed_dim=128,
                         depths=(2, 2, 18, 2),
                         num_heads=(4, 8, 16, 32),
                         decode_channels=decoder_channels)

    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model = XYCUNetFormer_1()
    #保存到txt文件
    with open('xycunetformermodel.txt', 'w') as f:
        print(model, file=f)
    #关闭文件
    f.close()
    print(model)