# EHE-DETR

English | [简体中文](README_CN.md)

**Small Object Detection in Aerial Images via Adaptive Hypergraph Reasoning and High-Resolution Feature Enhancement**

Built on DEIM / D-FINE, EHE-DETR combines efficient local feature extraction with adaptive hypergraph reasoning in parallel backbone branches. Shallow high-resolution features restore fine details while retaining three detection scales at strides 8, 16, and 32.

- **EFE**: Partial-channel spatial convolution and pointwise channel mixing for efficient local feature extraction.
- **AHR**: Adaptive regional prototypes and bidirectional hypergraph message passing for higher-order context modeling.
- **FHE**: Detail residuals from stride-4 features are encoded through a narrow branch and injected into P3 using PixelUnshuffle.

## Results

AP is averaged over IoU thresholds from 0.50 to 0.95. Accuracy is reported in %, and parameters in millions.

| Dataset | Split | Input | Model | AP50 | AP | Params (M) | GFLOPs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| VisDrone-DET2019 | val | 800×800 | D-FINE-S | 48.1 | 29.3 | 10.2 | 37.3 |
| VisDrone-DET2019 | val | 800×800 | **EHE-DETR-S** | **53.6** | **33.1** | 12.6 | 65.4 |
| AI-TODv2 | test | 800×800 | D-FINE-S | 54.1 | 24.7 | 10.2 | 37.3 |
| AI-TODv2 | test | 800×800 | **EHE-DETR-S** | **59.7** | **27.8** | 12.6 | 65.4 |
| TinyPerson | test | 1024×1024 | D-FINE-S | 30.7 | 11.8 | 10.2 | 59.3 |
| TinyPerson | test | 1024×1024 | **EHE-DETR-S** | **35.9** | **13.8** | 12.6 | 105.7 |

### Ablation Study

VisDrone-DET2019 validation set, 800×800 input.

| EFE | AHR | FHE | AP50 | AP |
| :---: | :---: | :---: | ---: | ---: |
| — | — | — | 48.1 | 29.3 |
| ✓ | — | — | 52.0 | 32.1 |
| — | ✓ | — | 50.8 | 31.0 |
| ✓ | ✓ | — | 53.1 | 32.6 |
| ✓ | ✓ | ✓ | **53.6** | **33.1** |

## Installation and Data

Experimental environment: Python 3.10, PyTorch 2.2.2, and an NVIDIA RTX 4090.

```bash
conda create -n ehe-detr python=3.10 -y
conda activate ehe-detr
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use COCO JSON annotations with consistent category mappings. Handle ignored regions and non-evaluation categories when converting VisDrone annotations.

```text
dataset/
├── visdrone/{train,val}/
├── aitod/{train,test}/
└── tinyperson/{train,test}/
```

Each split directory contains `images/` and `annotations.json`. Set the dataset root with `--data` and the evaluation split with `--split`.

## Training and Evaluation

Training uses 300 epochs, batch size 4, AdamW, a learning rate of 1e-4, weight decay of 1e-3, DEIM multi-stage augmentation, and EMA. Run commands from the repository root.

```bash
python experiment.py train --dataset visdrone --data /path/to/visdrone --device cuda:0
python experiment.py train --dataset aitod --data /path/to/aitodv2 --device cuda:0
python experiment.py train --dataset tinyperson --data /path/to/tinyperson --device cuda:0
```

The default variant is `full`, with outputs saved to `outputs/<dataset>/<variant>/`. Use `--output` for a separate experiment directory and `--seed` to set the random seed.

```bash
python experiment.py train --dataset visdrone --checkpoint outputs/visdrone/full/last.pth
python experiment.py eval --dataset visdrone --checkpoint outputs/visdrone/full/best_stg2.pth
python experiment.py infer --dataset visdrone --checkpoint outputs/visdrone/full/best_stg2.pth --input /path/to/images --output outputs/predictions
```

Use the same dataset and model configuration for training, evaluation, and inference. Select an available `best_stg1.pth` or `best_stg2.pth` checkpoint.

### Ablations and Module Comparisons

Add `--variant` to the training or evaluation command:

| Variant | Model |
| --- | --- |
| `baseline` | D-FINE-S |
| `efe` / `ahr` | EFE only / AHR only |
| `efe_ahr` / `full` | EFE+AHR / full model |
| `efe_c2f` / `efe_hgstage` | Replace the EFE branch with C2f / HG_Stage |
| `ahr_hypercompute` / `ahr_hyperace` | Replace the AHR branch with HyperCompute / HyperACE |

```bash
python experiment.py train --dataset visdrone --variant efe
python experiment.py train --dataset visdrone --variant full --edges 8
```

`--edges` supports values from 6 to 11; the default is 8.

### Utilities

```bash
python experiment.py prepare --dataset visdrone
python experiment.py smoke --device cpu
python experiment.py profile --checkpoint outputs/visdrone/full/best_stg2.pth --device cuda:0
python experiment.py summarize --output outputs
```

These commands export configurations, check forward and backward passes, measure speed, and aggregate metrics, respectively. Speed profiling measures model forward passes in FP32 with batch size 1, 10 warm-up iterations, and 30 timed iterations.

## Code Layout

| Component | Path |
| --- | --- |
| Model architectures | `configs/models/` |
| Dataset definitions | `configs/dataset/` |
| Training configurations | `configs/visdrone.yml`, `aitod.yml`, `tinyperson.yml` |
| EFE / AHR | `engine/extre_module/custom_nn/conv_module/efe_ahr.py` |
| FHE | `engine/extre_module/custom_nn/neck_module/fhe.py` |
| Module comparisons | `engine/extre_module/custom_nn/conv_module/efe_ahr_compare.py` |
| Model builder | `engine/extre_module/tasks.py` |

## Citation and Acknowledgements

Paper: **EHE-DETR: Small Object Detection in Aerial Images via Adaptive Hypergraph Reasoning and High-Resolution Feature Enhancement**.

Built on [DEIM](https://github.com/ShihuaHuang95/DEIM) / [D-FINE](https://github.com/Peterande/D-FINE).
