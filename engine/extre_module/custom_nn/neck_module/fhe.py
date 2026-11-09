import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.deim.utils import get_activation
__all__ = ['FHE']

class FHE(nn.Module):

    def __init__(self, c_detail, c_main, c_out, c_mid=16, act='silu', use_residual=True, use_gate=False):
        super().__init__()
        assert c_out == c_main, f'FHE 为残差注入，要求 c_out({c_out}) == c_main({c_main})'
        self.use_residual = use_residual
        self.use_gate = use_gate
        self.act = get_activation(act)
        self.squeeze = nn.Conv2d(c_detail, c_mid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.dw = nn.Conv2d(c_mid, c_mid, 3, 1, 1, groups=c_mid, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(c_mid * 4, c_out, 1, 1, 0, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if use_gate:
            self.gate = nn.Conv2d(c_main, 1, 1, 1, 0, bias=True)
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)

    @staticmethod
    def _high_freq(x):
        low = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        low = F.interpolate(low, size=x.shape[-2:], mode='nearest')
        return x - low

    def forward(self, x):
        (f_detail, p_main) = (x[0], x[1])
        d = self._high_freq(f_detail) if self.use_residual else f_detail
        d = self.act(self.bn1(self.squeeze(d)))
        d = self.act(self.bn2(self.dw(d)))
        (h, w) = d.shape[-2:]
        if h % 2 or w % 2:
            d = F.pad(d, [0, w % 2, 0, h % 2])
        d = self.unshuffle(d)
        delta = self.proj(d)
        if delta.shape[-2:] != p_main.shape[-2:]:
            delta = F.interpolate(delta, size=p_main.shape[-2:], mode='bilinear', align_corners=False)
        if self.use_gate:
            delta = delta * torch.sigmoid(self.gate(p_main))
        return p_main + delta
if __name__ == '__main__':
    torch.manual_seed(0)
    (B, Cd, Cm, H, W) = (2, 128, 256, 640, 640)
    f2 = torch.randn(B, Cd, H // 4, W // 4)
    p3 = torch.randn(B, Cm, H // 8, W // 8)
    m = FHE(Cd, Cm, Cm).eval()
    out = m([f2, p3])
    print(f'[shape ] F2={tuple(f2.shape)}  P3={tuple(p3.shape)}  ->  out={tuple(out.shape)}')
    err = (out - p3).abs().max().item()
    print(f"[init  ] identity-init max err = {err:.3e}  ->  {('OK' if err == 0 else 'FAIL')}")
    nn.init.normal_(m.proj.weight, std=0.02)
    delta = (m([f2, p3]) - p3).abs().max().item()
    print(f"[path  ] non-zero delta after re-init = {delta:.4f}  ->  {('OK' if delta > 0 else 'FAIL')}")
    const = torch.full((1, 8, 32, 32), 3.14)
    hf = FHE._high_freq(const).abs().max().item()
    print(f"[highfq] high-freq of constant map = {hf:.3e}  ->  {('OK' if hf < 1e-06 else 'FAIL')}")
    odd = FHE(Cd, Cm, Cm).eval()([torch.randn(1, Cd, 81, 81), torch.randn(1, Cm, 41, 41)])
    print(f'[odd   ] odd-size out = {tuple(odd.shape)} (expect (1, {Cm}, 41, 41))')
    n = sum((p.numel() for p in m.parameters()))
    print(f'[params] FHE = {n / 1000.0:.1f}K')
