import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'efe', 'ahr', 'efe_ahr', 'full', 'efe_c2f', 'efe_hgstage', 'ahr_hypercompute', 'ahr_hyperace')

def load_configuration(path):

    def merge(target, source):
        for (key, value) in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    path = Path(path)
    current = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    result = {}
    for include in current.pop('__include__', []):
        merge(result, load_configuration(path.parent / include))
    merge(result, current)
    return result

def configuration(args):
    cfg = load_configuration(ROOT / 'configs' / (args.dataset + '.yml'))
    if args.variant == 'baseline':
        baseline = load_configuration(ROOT / 'configs/baseline.yml')
        cfg.update({key: baseline[key] for key in ('DEIM', 'HGNetv2', 'HybridEncoder', 'DFINETransformer')})
        cfg['model'] = 'DEIM'
        cfg.pop('DEIM_MG', None)
    else:
        architecture = yaml.safe_load((ROOT / 'configs/models' / (args.variant + '.yaml')).read_text(encoding='utf-8'))
        if args.edges is not None:
            if args.variant not in ('ahr', 'efe_ahr', 'full'):
                raise ValueError('--edges applies to ahr/efe_ahr/full only')
            for (_, name, params) in architecture['backbone']:
                if name == 'EFEAHRBlock':
                    params[2] = args.edges
        model_path = args.output / 'architecture.yaml'
        model_path.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding='utf-8')
        cfg['DEIM_MG']['yaml_path'] = str(model_path)
    data = Path(args.data).resolve() if args.data else ROOT / 'dataset' / args.dataset
    for (loader, split) in [('train', 'train'), ('val', args.split or ('val' if args.dataset == 'visdrone' else 'test'))]:
        cfg[loader + '_dataloader']['dataset']['img_folder'] = str(data / split / 'images')
        cfg[loader + '_dataloader']['dataset']['ann_file'] = str(data / split / 'annotations.json')
    if args.batch:
        for split in ('train', 'val'):
            cfg[split + '_dataloader']['total_batch_size'] = args.batch
    cfg['output_dir'] = str(args.output)
    path = args.output / 'experiment.yml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return (path, cfg)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'eval', 'infer', 'profile', 'smoke', 'summarize'])
    p.add_argument('--dataset', choices=['visdrone', 'aitod', 'tinyperson'], default='visdrone')
    p.add_argument('--variant', choices=VARIANTS, default='full')
    p.add_argument('--edges', type=int, choices=range(6, 12))
    p.add_argument('--data', help='Root containing train/ and val/ or test/')
    p.add_argument('--split', help='Evaluation folder name')
    p.add_argument('--batch', type=int, help='Override only batch; does not auto-scale LR or EMA')
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--input', type=Path, help='Image or image folder for inference')
    p.add_argument('--threshold', type=float, default=0.25)
    p.add_argument('--size', type=int, help='Smoke/profile input size; defaults 64/real dataset size')
    p.add_argument('--iterations', type=int, default=30)
    args = p.parse_args()
    if args.action in ('eval', 'infer') and (not args.checkpoint):
        p.error('--checkpoint is required')
    if args.action == 'infer' and args.input is None:
        p.error('--input is required')
    args.output = (args.output or ROOT / 'outputs' / args.dataset / (args.variant + (f'_E{args.edges}' if args.edges else ''))).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (path, cfg) = configuration(args)
    print('Resolved configuration:', path)
    if args.action == 'prepare':
        return
    if args.action in ('train', 'eval'):
        if args.action == 'train' and (not args.checkpoint) and list(args.output.glob('*.pth')):
            raise FileExistsError('Existing checkpoints: choose a new --output or resume with --checkpoint')
        for split in ('train', 'val') if args.action == 'train' else ('val',):
            dataset = cfg[split + '_dataloader']['dataset']
            for key in ('img_folder', 'ann_file'):
                if not Path(dataset[key]).exists():
                    raise FileNotFoundError(f'{split}.{key}: {dataset[key]}')
        cmd = [sys.executable, str(ROOT / 'train.py'), '-c', str(path), '-d', args.device, '--seed', str(args.seed)]
        if args.checkpoint:
            cmd += ['-r', args.checkpoint]
        if args.action == 'eval':
            cmd += ['--test-only']
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if args.action == 'summarize':
        rows = []
        for log in args.output.rglob('log.txt'):
            for line in log.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                    if 'test_coco_eval_bbox' in row:
                        rows.append(dict(log=str(log), **row))
                except json.JSONDecodeError:
                    pass
        (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print('Wrote', len(rows), 'epoch records to metrics.json')
        return
    import torch
    from engine.core import YAMLConfig
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    config = YAMLConfig(str(path))
    model = config.model.to(args.device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu')
        state = state['ema']['module'] if state.get('ema') else state.get('model', state)
        model.load_state_dict(state, strict=True)
    size = args.size or (160 if args.action == 'smoke' else cfg['eval_spatial_size'][0])
    if args.action == 'smoke':
        for module in model.modules():
            if hasattr(module, 'eval_spatial_size'):
                module.eval_spatial_size = None
        x = torch.randn(1, 3, size, size, device=args.device)
        with torch.no_grad():
            result = model(x)
        assert torch.isfinite(result['pred_logits']).all()
        assert torch.isfinite(result['pred_boxes']).all()
        model.train()
        x = torch.randn(2, 3, size, size, device=args.device)
        targets = [{'labels': torch.tensor([0], device=args.device), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=args.device)} for _ in range(2)]
        loss = config.criterion.to(args.device)(model(x, targets), targets, epoch=0)
        total = sum(loss.values())
        assert torch.isfinite(total), loss
        total.backward()
        print('PASS forward / loss / backward; parameters:', sum((p.numel() for p in model.parameters())))
    elif args.action == 'profile':
        x = torch.randn(1, 3, size, size, device=args.device)
        if args.size:
            for module in model.modules():
                if hasattr(module, 'eval_spatial_size'):
                    module.eval_spatial_size = None

        def sync():
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
        with torch.no_grad():
            for _ in range(10):
                model(x)
            sync()
            start = time.perf_counter()
            for _ in range(args.iterations):
                model(x)
            sync()
            elapsed = time.perf_counter() - start
        report = dict(parameters=sum((p.numel() for p in model.parameters())), fps=args.iterations / elapsed, device=args.device, input_size=size, batch=1, warmup=10, iterations=args.iterations, precision='float32', includes_preprocessing=False, torch=torch.__version__)
        (args.output / 'profile.json').write_text(json.dumps(report, indent=2))
        print(report)
    else:
        from PIL import Image, ImageDraw
        from torchvision.transforms import functional as F
        files = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
        post = config.postprocessor.to(args.device).deploy()
        for file in files:
            if file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            image = Image.open(file).convert('RGB')
            tensor = F.to_tensor(image.resize(tuple(reversed(cfg['eval_spatial_size'])))).unsqueeze(0).to(args.device)
            with torch.no_grad():
                (labels, boxes, scores) = post(model(tensor), torch.tensor([image.size], device=args.device))
            draw = ImageDraw.Draw(image)
            for (label, box, score) in zip(labels[0], boxes[0], scores[0]):
                if score < args.threshold:
                    continue
                coords = box.cpu().tolist()
                draw.rectangle(coords, outline='red', width=2)
                draw.text(coords[:2], f'{int(label)} {float(score):.2f}', fill='red')
            image.save(args.output / (file.stem + '_prediction.png'))
if __name__ == '__main__':
    main()
