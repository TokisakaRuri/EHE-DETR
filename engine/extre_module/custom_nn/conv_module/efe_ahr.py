import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...ultralytics_nn.conv import Conv
__all__ = ['EFEAHR', 'EFEAHRBlock']

class PConv(nn.Module):

    def __init__(self, dim, n_div=4, k=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k // 2, bias=False)

    def forward(self, x):
        if self.dim_conv3 == 0:
            return x
        (x1, x2) = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)

class EFEBlock(nn.Module):

    def __init__(self, c_in, c_out=None, k=3, dropout=0.0):
        super().__init__()
        c_out = c_out or c_in
        self.pconv = PConv(c_in, 4, k)
        hidden_dim = c_in * 2
        self.pw_conv1 = Conv(c_in, hidden_dim, 1)
        self.pw_conv2 = nn.Conv2d(hidden_dim, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.pconv(x)
        x = self.pw_conv1(x)
        x = self.bn2(self.pw_conv2(x))
        x = self.drop(x)
        return identity + x

class EFE(nn.Module):

    def __init__(self, c_in, c_out, n=1, dropout=0.0):
        super().__init__()
        self.conv_entry = Conv(c_in, c_out, 1)
        self.blocks = nn.Sequential(*(EFEBlock(c_out, c_out, dropout=dropout) for _ in range(n)))

    def forward(self, x):
        x = self.conv_entry(x)
        return self.blocks(x)

class LowRankLinear(nn.Module):

    def __init__(self, in_features, out_features, rank=8, bias=True):
        super().__init__()
        r = min(rank, in_features // 2 if in_features > 1 else 1, out_features // 2 if out_features > 1 else 1)
        r = max(1, r)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=bias)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))

class LightFFN(nn.Module):

    def __init__(self, embed_dim, hidden_dim=None, dropout=0.1, act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.w1 = nn.Linear(embed_dim, hidden_dim)
        self.act = act_layer()
        self.w2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))

class AnchorHyperedgeGen(nn.Module):

    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, lora_rank=8):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.grid_size = int(math.sqrt(num_hyperedges))
        self.query_proj = LowRankLinear(node_dim, node_dim // 2, rank=lora_rank)
        self.key_proj = LowRankLinear(node_dim, node_dim // 2, rank=lora_rank)
        self.bias_table = nn.Parameter(torch.zeros(num_heads, num_hyperedges))
        self.scale = (node_dim // 2 // num_heads) ** (-0.5)

    def forward(self, X, H, W):
        (B, N, D) = X.shape
        X_img = X.transpose(1, 2).view(B, D, H, W)
        anchors = F.adaptive_avg_pool2d(X_img, (self.grid_size, self.grid_size))
        anchors = anchors.flatten(2).transpose(1, 2)
        if anchors.shape[1] != self.num_hyperedges:
            anchors = F.interpolate(anchors.transpose(1, 2), size=self.num_hyperedges, mode='linear').transpose(1, 2)
        Q = self.query_proj(X).view(B, N, self.num_heads, -1).transpose(1, 2)
        K = self.key_proj(anchors).view(B, self.num_hyperedges, self.num_heads, -1).permute(0, 2, 3, 1)
        attn_logits = Q @ K * self.scale
        attn_logits = attn_logits + self.bias_table.unsqueeze(0).unsqueeze(2)
        return attn_logits

class SymmetricAdaHGConv(nn.Module):

    def __init__(self, embed_dim, num_hyperedges, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.edge_generator = AnchorHyperedgeGen(embed_dim, num_hyperedges, num_heads, dropout, lora_rank=8)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(), nn.Dropout(dropout))
        self.node_proj = LightFFN(embed_dim, dropout=dropout)
        self.edge_gate = nn.Sequential(nn.Linear(embed_dim, embed_dim // 4), nn.SiLU(), nn.Linear(embed_dim // 4, num_heads), nn.Sigmoid())

    def forward(self, X, H, W):
        (B, N, D) = X.shape
        attn_logits = self.edge_generator(X, H, W)
        A_N2E = F.softmax(attn_logits, dim=2)
        X_heads = X.view(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        He_heads = torch.matmul(A_N2E.transpose(2, 3), X_heads)
        He = He_heads.transpose(1, 2).reshape(B, -1, D)
        He = self.edge_proj(He)
        A_E2N = F.softmax(attn_logits, dim=3)
        gate = self.edge_gate(He).view(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        He_heads_new = He.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)
        He_gated = He_heads_new * gate
        X_out_heads = torch.matmul(A_E2N, He_gated)
        X_out = X_out_heads.transpose(1, 2).reshape(B, N, D)
        return X + self.node_proj(X_out)

class EfficientHGCompute(nn.Module):

    def __init__(self, embed_dim, num_hyperedges, num_heads, dropout=0.1):
        super().__init__()
        self.hgnn = SymmetricAdaHGConv(embed_dim, num_hyperedges, num_heads, dropout)

    def forward(self, x):
        (B, C, H, W) = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.hgnn(tokens, H, W)
        return tokens.transpose(1, 2).view(B, C, H, W)

class AHR(nn.Module):

    def __init__(self, c_in, c_out, num_hyperedges=8, dropout=0.1):
        super().__init__()
        c_mid = c_in // 2 if c_in > 32 else c_in
        num_heads = max(1, c_mid // 16)
        if c_mid % num_heads != 0:
            num_heads = 1
        self.pre_conv = Conv(c_in, c_mid, 1)
        self.hgnn = EfficientHGCompute(c_mid, num_hyperedges, num_heads, dropout)
        self.post_conv = Conv(c_mid, c_out, 1)
        self.shortcut = Conv(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.pre_conv(x)
        y = self.hgnn(y)
        y = self.post_conv(y)
        return self.shortcut(x) + y

class AHRCompatibilityBlock(AHR):
    pass

class EFEAHRBlock(nn.Module):

    def __init__(self, c_in, c_out, n=1, num_hyperedges=8, dropout=0.1, use_EFE=True, use_AHR=True):
        super().__init__()
        if not (use_EFE or use_AHR):
            raise ValueError('At least one of use_EFE/use_AHR must be True.')
        self.use_EFE = use_EFE
        self.use_AHR = use_AHR
        if self.use_EFE:
            self.baseline_path = EFE(c_in, c_out, n=n, dropout=dropout)
        if self.use_AHR:
            self.expert_path = AHR(c_in, c_out, num_hyperedges=num_hyperedges, dropout=dropout)

    def forward(self, x):
        outs = []
        if self.use_EFE:
            outs.append(self.baseline_path(x))
        if self.use_AHR:
            outs.append(self.expert_path(x))
        if len(outs) == 1:
            return outs[0]
        return sum(outs)

class EFEAHR(nn.Module):

    def __init__(self, c_in, c_out, n=1, e_mid=0.5, num_hyperedges=8, dropout=0.1):
        super().__init__()
        self.baseline_path = EFE(c_in, c_out, n=n, dropout=dropout)
        self.expert_path = AHR(c_in, c_out, num_hyperedges=num_hyperedges, dropout=dropout)

    def forward(self, x):
        return self.baseline_path(x) + self.expert_path(x)
