# EHE-DETR

**基于自适应超图推理与高分辨率特征增强的航拍图像小目标检测**

面向 GitHub 发布的独立精简实现，基于 DEIM / D-FINE 工程，包含训练、评估、检测可视化、消融、模块替换与超边数量实验。模块命名：**EFE、AHR、FHE**。

EFE 对部分通道进行空间卷积；AHR 通过输入自适应原型与双向超图消息传递建模上下文，两者在主干中并行相加；FHE 从 stride-4 特征提取细节残差，经窄通道编码与 PixelUnshuffle 注入 P3。检测器仍输出 stride 8/16/32 三尺度特征。

## 主要结果

以下为**论文报告结果，并非编写本 README 时重新训练或测速所得**。AP 表示 IoU=0.50:0.95 的平均精度，AP50 表示 IoU=0.50 的平均精度，精度单位为 %。

| 数据集 | 评估集 | 输入尺寸 | 模型 | AP50 | AP | Params (M) | GFLOPs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| VisDrone-DET2019 | val | 800×800 | D-FINE-S | 48.1 | 29.3 | 10.2 | 37.3 |
| VisDrone-DET2019 | val | 800×800 | **EHE-DETR** | **53.6** | **33.1** | 12.6 | 65.4 |
| AI-TODv2 | test | 800×800 | D-FINE-S | 54.1 | 24.7 | 10.2 | 37.3 |
| AI-TODv2 | test | 800×800 | **EHE-DETR** | **59.7** | **27.8** | 12.6 | 65.4 |
| TinyPerson | test | 1024×1024 | D-FINE-S | 30.7 | 11.8 | 10.2 | 59.3 |
| TinyPerson | test | 1024×1024 | **EHE-DETR** | **35.9** | **13.8** | 12.6 | 88.1 |

相较于 D-FINE-S，三个数据集上的 AP50 分别提升 **5.5、5.6、5.2 个百分点**，AP 分别提升 **3.8、3.1、2.0 个百分点**。完整模型的参数量与 GFLOPs 高于基线。

### 消融实验

VisDrone-DET2019 验证集，800×800 输入，数据来自论文表 5。

| EFE | AHR | FHE | AP50 | AP | Params (M) | GFLOPs | FPS |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| — | — | — | 48.1 | 29.3 | 10.2 | 37.3 | 69.3 |
| ✓ | — | — | 52.0 | 32.1 | 11.7 | 59.0 | 142.1 |
| — | ✓ | — | 50.8 | 31.0 | 9.9 | 44.1 | 115.2 |
| ✓ | ✓ | — | 53.1 | 32.6 | 12.6 | 65.4 | 87.7 |
| ✓ | ✓ | ✓ | **53.6** | **33.1** | 12.6 | 65.4 | 92.4 |

参数量和 GFLOPs 按论文精度取舍，相同显示值不代表 FHE 没有开销。FPS 为论文报告值；测速精度、预热、batch size 及是否包含预处理等细节尚需随正式实验记录补充。

## 安装

论文环境：Python 3.10、PyTorch 2.2.2、RTX 4090。建议在 Linux/CUDA 环境运行完整训练。

```bash
conda create -n ehe-detr python=3.10 -y
conda activate ehe-detr
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

本包去除了原工程中的 CLIP、Mamba、无关注意力模块、MMCV 等导入。保留的是实际训练与对比模块的依赖链，不需要原工程目录。当前检查环境为 PyTorch 2.12.0+cpu，论文环境的 GPU 训练尚待执行验证。

## 数据准备

自行获取数据并按数据集协议转换为 COCO JSON。图像与标注采用以下组织，`annotations.json` 包含 images、annotations、categories；category_id 与类别顺序须保持一致。

```text
dataset/
├── visdrone/{train,val}/
│   ├── images/
│   └── annotations.json
├── aitod/{train,test}/
│   ├── images/
│   └── annotations.json
└── tinyperson/{train,test}/
    ├── images/
    └── annotations.json
```

VisDrone 为 10 类，使用 val 评估；AI-TODv2 为 8 类，TinyPerson 为 1 类，使用 test 评估。转换 VisDrone 时须处理忽略区域及非评估类别。本包不重新划分数据，不附带数据集或转换结果。需要验证集选择超参数时，可通过 `--split val` 切换已有划分。

## 快速开始

从本包根目录运行。先检查数据路径和合并后的训练设置：

```bash
python experiment.py prepare --dataset visdrone --data /path/to/visdrone
python experiment.py smoke --device cpu
python experiment.py train --dataset visdrone --data /path/to/visdrone --device cuda:0 --seed 0
```

`prepare` 只生成 `outputs/visdrone/full/experiment.yml` 与结构快照，不读取数据。`smoke` 使用随机 160×160 图像，检查前向、匹配、损失和反向传播，不代表精度复现。种子 0 为示例；论文文本未说明种子。

训练配置为 300 epochs、AdamW，VisDrone/TinyPerson 学习率 1e-4、weight decay 0.001、batch=4，继承 DEIM 多阶段增强、EMA 与调度策略。VisDrone/AI-TODv2 输入 800，TinyPerson 输入 1024。完整解析设置保存在 experiment.yml；原配置中的分阶段增强时间点也予以保留。

训练前请确认输出目录中没有需要保留的旧实验；重新训练应指定新的 `--output`，命令入口会拒绝在已有 checkpoint 的目录中从头训练。断点续训使用同一模型和数据设置：

```bash
python experiment.py train --checkpoint outputs/visdrone/full/last.pth --device cuda:0
python experiment.py eval --checkpoint outputs/visdrone/full/best_stg2.pth --device cuda:0
python experiment.py infer --checkpoint outputs/visdrone/full/best_stg2.pth --input /path/to/images --output outputs/predictions --device cuda:0
```

权重由训练生成，选择实际存在的 `best_stg1.pth` 或 `best_stg2.pth`。推理输出类别编号与置信度，编号按当前训练类别映射解释。新包统一模块名称，保留参数属性名，已有权重需使用匹配结构，以严格加载结果为准。

## 完整实验流程

统一入口通过 `--dataset`、`--variant` 和 `--edges` 组合实验，避免为每个组合复制训练文件。每个实验默认使用独立输出目录。

| 实验 | 参数 |
| --- | --- |
| 原始 D-FINE-S | `--variant baseline` |
| 仅 EFE | `--variant efe` |
| 仅 AHR | `--variant ahr` |
| EFE+AHR | `--variant efe_ahr` |
| 完整 EHE-DETR | `--variant full`（默认） |
| EFE 分支替换为 C2f | `--variant efe_c2f` |
| EFE 分支替换为 HG_Stage | `--variant efe_hgstage` |
| AHR 分支替换为 HyperCompute | `--variant ahr_hypercompute` |
| AHR 分支替换为 HyperACE | `--variant ahr_hyperace` |
| 超边数实验 | `--variant full --edges 6`，取值 6–11 |

例如：

```bash
python experiment.py train --dataset visdrone --variant baseline
python experiment.py train --dataset visdrone --variant efe
python experiment.py train --dataset visdrone --variant ahr
python experiment.py train --dataset visdrone --variant efe_ahr
python experiment.py train --dataset visdrone --variant full --edges 8
python experiment.py train --dataset visdrone --variant ahr_hyperace
python experiment.py train --dataset aitod --variant full --data /path/to/aitodv2
python experiment.py train --dataset tinyperson --variant full --data /path/to/tinyperson
```

AI-TOD 统一使用 batch=4、lr=1e-4 的配置；该配置依据论文与现有公共训练设置整理，尚待真实数据训练复现。

多卡可在生成配置后调用原训练入口：

```bash
python experiment.py prepare --dataset visdrone
torchrun --standalone --nproc_per_node=2 train.py -c outputs/visdrone/full/experiment.yml
```

`total_batch_size` 为所有卡合计，需能被 GPU 数整除。Windows 下优先使用单卡；多卡训练需匹配分布式后端。

## 速度与日志

```bash
python experiment.py profile --checkpoint outputs/visdrone/full/best_stg2.pth --device cuda:0
python experiment.py summarize --output outputs
```

`profile` 保存 profile.json：FP32、batch=1、预热 10 次、默认计时 30 次，GPU 同步，不计预处理与后处理。没有 checkpoint 时测量随机初始化模型。该口径与论文 FPS 是否相同尚未确认，不应据此声称复现论文速度。训练器保留原 GFLOPs 统计入口；`summarize` 汇总输出目录下 log.txt 的 COCO 指标到 metrics.json，记录顺序 AP、AP50、AP75 等遵循原评估器。

数据集定义放在 `configs/dataset/`，各数据集训练配置通过 `__include__` 引用 `configs/dfine/` 与 `configs/base/`。模型结构文件保留源代码的 `[from, module, args]` 逐层写法与分组，移除了代码注释。

## 文件结构

```text
EHE-DETR/
├── experiment.py
├── train.py
├── configs/
├── engine/
├── requirements.txt
├── SOURCE_MANIFEST.json
├── THIRD_PARTY_NOTICES.md
├── VERIFICATION.md
└── LICENSE
```

创新模块位于：

- `engine/extre_module/custom_nn/conv_module/efe_ahr.py`：EFE/AHR 与并行主干。
- `engine/extre_module/custom_nn/neck_module/fhe.py`：FHE。
- `engine/extre_module/tasks.py`：精简图结构解析与模型构建。
- `engine/extre_module/custom_nn/conv_module/efe_ahr_compare.py`：模块替换对比。

保留原训练、损失、匹配、数据增强、EMA、调度和 COCO 评估实现。去掉蒸馏入口与无关注册项，conv/block 文件仅提取实际所需定义。Python 注释与文档字符串、YAML 注释已删除；原版权与许可声明集中保存在 SOURCE_COPYRIGHT_NOTICES.md。SOURCE_MANIFEST.json 记录来源文件。完整 CPU 检查范围见 VERIFICATION.md。

## 引用与致谢

论文：EHE-DETR：基于自适应超图推理与高分辨率特征增强的航拍图像小目标检测。作者、论文链接及正式引用信息待补充。

基于 [DEIM](https://github.com/ShihuaHuang95/DEIM) / [D-FINE](https://github.com/Peterande/D-FINE)。说明文档结构参考 [EduYOLO](https://github.com/datadance/EduYOLO)，感谢相关模块与数据集作者。

## 许可证

LICENSE 保留上游 Apache-2.0 文本；部分集成文件包含 Ultralytics AGPL-3.0 声明，不能把整个整合包简单表述为仅 Apache-2.0。第三方来源与已发现的许可声明见 THIRD_PARTY_NOTICES.md，发布时需保留对应声明。
