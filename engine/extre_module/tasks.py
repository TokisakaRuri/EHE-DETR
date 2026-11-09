import yaml
import ast
from pathlib import Path
import torch
from torch import nn
from ..core import register
from ..backbone.common import FrozenBatchNorm2d
from ..backbone.hgnetv2 import StemBlock, HG_Stage
from ..deim.hybrid_encoder import RepNCSPELAN4, ConvNormLayer_fuse, SCDown, TransformerEncoderBlock
from ..deim.dfine_decoder import DFINETransformer
from .ultralytics_nn.conv import Conv, Concat
from .custom_nn.conv_module.efe_ahr import EFEAHRBlock
from .custom_nn.conv_module.efe_ahr_compare import EFEAHRCompare
from .custom_nn.neck_module.fhe import FHE

@register(force=True)
class DEIM_MG(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, yaml_path, num_classes=80, eval_spatial_size=(640, 640), pretrained=None):
        super().__init__()
        if pretrained:
            raise ValueError('Use train.py -t for explicit checkpoint initialization')
        d = yaml.safe_load(Path(yaml_path).read_text(encoding='utf-8'))
        (self.backbone, self.encoder, self.decoder, self.save) = parse_model(d, 3, num_classes, eval_spatial_size)

    def forward(self, x, targets=None):
        y = []
        for (idx, m) in enumerate(list(self.backbone.children()) + list(self.encoder.children())):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        x = self.decoder([y[j] for j in self.decoder.f], targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for (name, child) in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

def parse_model(d, channels, nc, eval_spatial_size):
    (ch, layers, save) = ([], [], [])
    groups = []
    for section in ('backbone', 'encoder', 'decoder'):
        group = []
        for (f, name, args) in d[section]:
            i = len(layers)
            module = getattr(nn, name[3:]) if name.startswith('nn.') else globals()[name]

            def width(j):
                return ch[j] if ch else channels
            if module is DFINETransformer:
                args = dict(args, feat_channels=[width(j) for j in f], num_classes=nc, eval_spatial_size=eval_spatial_size)
                (obj, c2) = (module(**args), width(f[0]))
            else:
                args = list(args)
                for (j, value) in enumerate(args):
                    if isinstance(value, str):
                        try:
                            args[j] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pass
                c2 = width(f) if isinstance(f, int) else width(f[-1])
                if module in (Conv, ConvNormLayer_fuse, SCDown, RepNCSPELAN4, EFEAHRBlock, EFEAHRCompare):
                    c2 = args[0]
                    args = [width(f), *args]
                elif module in (StemBlock, HG_Stage):
                    c2 = args[1]
                    args = [width(f), *args]
                elif module is TransformerEncoderBlock:
                    args = [width(f), *args]
                elif module is Concat:
                    c2 = sum((width(j) for j in f))
                elif module is FHE:
                    c2 = args[0]
                    args = [width(f[0]), width(f[1]), *args]
                obj = module(*args)
            (obj.i, obj.f, obj.type) = (i, f, name)
            obj.np = sum((p.numel() for p in obj.parameters()))
            save.extend((j % i for j in ([f] if isinstance(f, int) else f) if j != -1))
            layers.append(obj)
            group.append(obj)
            ch.append(c2)
        groups.append(group)
    return (nn.Sequential(*groups[0]), nn.Sequential(*groups[1]), groups[2][0], sorted(save))
