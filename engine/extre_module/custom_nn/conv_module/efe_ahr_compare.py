import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ultralytics_nn.conv import Conv
from ...ultralytics_nn.block import C2f
from engine.backbone.hgnetv2 import HG_Stage
from ..neck_module.HyperCompute import HyperComputeModule
from ..neck_module.HyperACE import HyperACE
from .efe_ahr import EFE, EfficientHGCompute
__all__ = ['EFEAHRCompare']

def build_EFE_path(efe_type, c_in, c_out, n=1, dropout=0.1, hg_layer_num=3):
    efe_type = efe_type.lower()
    if efe_type == 'faster':
        return EFE(c_in, c_out, n=n, dropout=dropout)
    if efe_type == 'hgstage':
        return HG_Stage(in_chs=c_in, mid_chs=max(16, c_out // 4), out_chs=c_out, block_num=n, layer_num=hg_layer_num, downsample=False, light_block=False, kernel_size=3, use_lab=False, agg='se', drop_path=0.0)
    if efe_type == 'c2f':
        return C2f(c_in, c_out, n=n, shortcut=True)
    raise ValueError(f'Unknown efe_type: {efe_type}')

class HyperComputeModule_Pooled(HyperComputeModule):

    def __init__(self, c, threshold, max_grid=40):
        super().__init__(c, threshold)
        self.max_grid = max_grid

    def forward(self, x):
        (b, c, h, w) = x.shape
        if h * w <= self.max_grid ** 2:
            return super().forward(x)
        (gh, gw) = (min(h, self.max_grid), min(w, self.max_grid))
        xp = F.adaptive_avg_pool2d(x, (gh, gw))
        t = xp.view(b, c, -1).transpose(1, 2).contiguous()
        distance = torch.cdist(t, t)
        hg = (distance < self.threshold).to(t.dtype)
        msg = self.hgconv(t, hg).to(t.dtype)
        msg = msg.transpose(1, 2).contiguous().view(b, c, gh, gw)
        msg = F.interpolate(msg, size=(h, w), mode='nearest')
        return self.act(self.bn(x + msg))

class HyperACE_Single(HyperACE):

    def __init__(self, c1, c2, n=1, num_hyperedges=8, dsc3k=True, shortcut=False, e1=0.5, e2=1, context='both'):
        super().__init__(c1, c2, n=n, num_hyperedges=num_hyperedges, dsc3k=dsc3k, shortcut=shortcut, e1=e1, e2=e2, context=context, channel_adjust=False)
        self.fuse = Conv(c1, c1, 1)

class CompareExpertBlock(nn.Module):

    def __init__(self, c_in, c_out, ahr_type='hgnn', num_hyperedges=8, dropout=0.1, hyper_threshold=10.0, hyper_max_grid=40, hyperace_e1=0.5):
        super().__init__()
        ahr_type = ahr_type.lower()
        c_mid = c_in // 2 if c_in > 32 else c_in
        if ahr_type == 'hyperace':
            unit = int(round(16 / hyperace_e1))
            c_mid = max(unit, c_mid // unit * unit)
        self.pre_conv = Conv(c_in, c_mid, 1)
        if ahr_type == 'hgnn':
            num_heads = max(1, c_mid // 16)
            if c_mid % num_heads != 0:
                num_heads = 1
            self.core = EfficientHGCompute(c_mid, num_hyperedges, num_heads, dropout)
        elif ahr_type == 'hypercompute':
            self.core = HyperComputeModule_Pooled(c_mid, hyper_threshold, max_grid=hyper_max_grid)
        elif ahr_type == 'hyperace':
            self.core = HyperACE_Single(c_mid, c_mid, n=1, num_hyperedges=num_hyperedges, e1=hyperace_e1)
        else:
            raise ValueError(f'Unknown ahr_type: {ahr_type}')
        self.post_conv = Conv(c_mid, c_out, 1)
        self.shortcut = Conv(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.pre_conv(x)
        y = self.core(y)
        y = self.post_conv(y)
        return self.shortcut(x) + y

class EFEAHRCompare(nn.Module):

    def __init__(self, c_in, c_out, n=1, num_hyperedges=8, dropout=0.1, efe_type='faster', ahr_type='hgnn', hg_layer_num=3, hyper_threshold=10.0, hyper_max_grid=40, hyperace_e1=0.5):
        super().__init__()
        (efe_type, ahr_type) = (str(efe_type).lower(), str(ahr_type).lower())
        self.use_EFE = efe_type != 'none'
        self.use_AHR = ahr_type != 'none'
        if not (self.use_EFE or self.use_AHR):
            raise ValueError('At least one of efe_type/ahr_type must not be "none".')
        if self.use_EFE:
            self.baseline_path = build_EFE_path(efe_type, c_in, c_out, n=n, dropout=dropout, hg_layer_num=hg_layer_num)
        if self.use_AHR:
            self.expert_path = CompareExpertBlock(c_in, c_out, ahr_type=ahr_type, num_hyperedges=num_hyperedges, dropout=dropout, hyper_threshold=hyper_threshold, hyper_max_grid=hyper_max_grid, hyperace_e1=hyperace_e1)

    def forward(self, x):
        outs = []
        if self.use_EFE:
            outs.append(self.baseline_path(x))
        if self.use_AHR:
            outs.append(self.expert_path(x))
        return outs[0] if len(outs) == 1 else sum(outs)
if __name__ == '__main__':
    x = torch.randn(1, 64, 40, 40)
    for (a, c) in [('hgstage', 'none'), ('c2f', 'none'), ('none', 'hypercompute'), ('none', 'hyperace'), ('faster', 'hgnn')]:
        m = EFEAHRCompare(64, 128, 1, 8, 0.1, a, c)
        print(a, c, m(x).shape, sum((p.numel() for p in m.parameters())))
