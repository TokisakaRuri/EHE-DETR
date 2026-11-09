# Verification record

Date: 2026-11-06 (Asia/Shanghai)
Environment: Windows, Python 3.10.11, PyTorch 2.12.0+cpu.

Passed:

- Python AST syntax checks for all packaged Python files.
- Standalone engine import and full EHE-DETR configuration construction.
- Eight architecture files compared with the original graph parser using the same random seed: identical state_dict keys, initialized tensor values, and saved graph references.
- EHE-DETR and original D-FINE-S: random-image inference, finite detection outputs, Hungarian matching, criterion losses and backward propagation at 160x160. Parameter counts: EHE-DETR 12,682,809; baseline 10,231,997.
- Seven additional ablation/replacement variants: finite forward outputs and shape (1,300,4).
- Hyperedge settings E=6 through E=11: finite forward outputs.
- Strict random checkpoint loading and image prediction export via experiment.py infer.

Scope limits:

No real dataset training, GPU timing, full-epoch dataloader run, or paper accuracy reproduction was performed. The pinned PyTorch 2.2.2 CUDA installation was not recreated. CPU checks establish graph and loss execution in the installed environment, not reproduction of reported accuracy. Multi-GPU execution remains untested. The AI-TOD batch4 configuration is a reconstruction from paper/common configuration settings, not an authenticated archived training configuration.

The release archive excludes temporary checkpoints, generated configurations, test images, output folders, datasets and Python caches.

Configuration layout update: source-style model YAML and layered D-FINE/DEIM includes restored. All three dataset configurations resolve identically through the CLI and engine loaders (excluding include metadata). Full-model forward/loss/backward rechecked. AI-TOD uses batch4 only.

Naming/comment update: EFE, AHR and FHE classes, efe/ahr/efe_ahr/full variant names, synchronized imports and YAML references. All 8 variants passed 160x160 CPU forward checks; full passed criterion/backward. All packaged Python files parsed successfully and contain no COMMENT tokens. Copyright/license notices moved to SOURCE_COPYRIGHT_NOTICES.md. Parameter attribute names remain unchanged for state_dict compatibility.
