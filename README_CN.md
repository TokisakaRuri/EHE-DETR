# EHE-DETR

[English](README.md) | 简体中文

**基于自适应超图推理与高分辨率特征增强的航拍图像小目标检测**

EHE-DETR 基于 DEIM / D-FINE，通过并行的高效特征提取与自适应超图推理增强主干，并利用浅层高分辨率特征补充小目标细节，保留 stride 8/16/32 三尺度检测结构。

- **EFE**：部分通道空间卷积与逐点通道交互，高效提取局部特征。
- **AHR**：自适应区域原型与双向超图消息传递，建模高阶上下文。
- **FHE**：提取 stride-4 特征的细节残差，经窄通道编码与 PixelUnshuffle 注入 P3。

## 实验结果

AP 为 IoU=0.50:0.95 的平均精度，精度单位为 %，参数量单位为 M。

| 数据集 | 划分 | 输入 | 模型 | AP50 | AP | 参数量 | GFLOPs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| VisDrone-DET2019 | val | 800×800 | D-FINE-S | 48.1 | 29.3 | 10.2 | 37.3 |
| VisDrone-DET2019 | val | 800×800 | **EHE-DETR-S** | **53.6** | **33.1** | 12.6 | 65.4 |
| AI-TODv2 | test | 800×800 | D-FINE-S | 54.1 | 24.7 | 10.2 | 37.3 |
| AI-TODv2 | test | 800×800 | **EHE-DETR-S** | **59.7** | **27.8** | 12.6 | 65.4 |
| TinyPerson | test | 1024×1024 | D-FINE-S | 30.7 | 11.8 | 10.2 | 59.3 |
| TinyPerson | test | 1024×1024 | **EHE-DETR-S** | **35.9** | **13.8** | 12.6 | 105.7 |

### 消融实验

VisDrone-DET2019 验证集，输入 800×800。

| EFE | AHR | FHE | AP50 | AP |
| :---: | :---: | :---: | ---: | ---: |
| — | — | — | 48.1 | 29.3 |
| ✓ | — | — | 52.0 | 32.1 |
| — | ✓ | — | 50.8 | 31.0 |
| ✓ | ✓ | — | 53.1 | 32.6 |
| ✓ | ✓ | ✓ | **53.6** | **33.1** |

## 安装与数据

实验环境：Python 3.10、PyTorch 2.2.2、RTX 4090。

```bash
conda create -n ehe-detr python=3.10 -y
conda activate ehe-detr
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

数据采用 COCO JSON 格式，类别映射须与标注一致。VisDrone 转换时需处理忽略区域及非评估类别。

```text
dataset/
├── visdrone/{train,val}/
├── aitod/{train,test}/
└── tinyperson/{train,test}/
```

每个划分目录包含 `images/` 和 `annotations.json`。使用 `--data` 指定数据集根目录，`--split` 切换评估划分。

## 训练与评估

训练设置：300 epochs、batch=4、AdamW、lr=1e-4、weight decay=1e-3，采用 DEIM 多阶段增强与 EMA。以下命令从仓库根目录执行。

```bash
python experiment.py train --dataset visdrone --data /path/to/visdrone --device cuda:0
python experiment.py train --dataset aitod --data /path/to/aitodv2 --device cuda:0
python experiment.py train --dataset tinyperson --data /path/to/tinyperson --device cuda:0
```

默认模型为 `full`，输出目录为 `outputs/<dataset>/<variant>/`。通过 `--output` 指定独立实验目录，`--seed` 设置随机种子。

```bash
python experiment.py train --dataset visdrone --checkpoint outputs/visdrone/full/last.pth
python experiment.py eval --dataset visdrone --checkpoint outputs/visdrone/full/best_stg2.pth
python experiment.py infer --dataset visdrone --checkpoint outputs/visdrone/full/best_stg2.pth --input /path/to/images --output outputs/predictions
```

评估和推理使用与训练一致的数据集及模型配置；权重可选择实际生成的 `best_stg1.pth` 或 `best_stg2.pth`。

### 消融与模块对比

在训练或评估命令中添加 `--variant`：

| 配置 | 模型 |
| --- | --- |
| `baseline` | D-FINE-S |
| `efe` / `ahr` | 仅 EFE / 仅 AHR |
| `efe_ahr` / `full` | EFE+AHR / 完整模型 |
| `efe_c2f` / `efe_hgstage` | EFE 分支替换为 C2f / HG_Stage |
| `ahr_hypercompute` / `ahr_hyperace` | AHR 分支替换为 HyperCompute / HyperACE |

```bash
python experiment.py train --dataset visdrone --variant efe
python experiment.py train --dataset visdrone --variant full --edges 8
```

`--edges` 支持 6–11，默认超边数为 8。

### 工具

```bash
python experiment.py prepare --dataset visdrone
python experiment.py smoke --device cpu
python experiment.py profile --checkpoint outputs/visdrone/full/best_stg2.pth --device cuda:0
python experiment.py summarize --output outputs
```

依次用于配置导出、前向与反向检查、速度测试和指标汇总。速度测试采用 FP32、batch=1、预热 10 次、计时 30 次，仅统计模型前向。

## 代码位置

| 内容 | 路径 |
| --- | --- |
| 模型结构 | `configs/models/` |
| 数据集定义 | `configs/dataset/` |
| 数据集训练配置 | `configs/visdrone.yml`、`aitod.yml`、`tinyperson.yml` |
| EFE / AHR | `engine/extre_module/custom_nn/conv_module/efe_ahr.py` |
| FHE | `engine/extre_module/custom_nn/neck_module/fhe.py` |
| 模块替换 | `engine/extre_module/custom_nn/conv_module/efe_ahr_compare.py` |
| 模型构建 | `engine/extre_module/tasks.py` |

## 引用与致谢

论文：**EHE-DETR：基于自适应超图推理与高分辨率特征增强的航拍图像小目标检测**。

基于 [DEIM](https://github.com/ShihuaHuang95/DEIM) / [D-FINE](https://github.com/Peterande/D-FINE)。
