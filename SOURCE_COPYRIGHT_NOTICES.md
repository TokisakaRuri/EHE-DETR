# Retained source notices

Source code comments and docstrings were removed for this distribution. Copyright and license notices extracted from those files are retained here; see also THIRD_PARTY_NOTICES.md and LICENSE. Paths below identify original release filenames before module renaming.

## train.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

## engine/__init__.py

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     

## engine/backbone/common.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)    
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py   
BatchNorm2d where the batch statistics and the affine parameters are fixed.
Copy-paste from torchvision.misc.ops with added eps before rqsrt,
without which any other models than torchvision.models.resnet[18,34,50,101]     
produce nans.

## engine/backbone/hgnetv2.py

reference 
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

## engine/core/workspace.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)   
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/core/yaml_config.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)   
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/core/yaml_utils.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)   
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/core/_config.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/core/__init__.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/data/dataloader.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.   

## engine/data/_misc.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.    

## engine/data/__init__.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR) 
Copyright(c) 2023 lyuwenyu. All Rights Reserved.  

## engine/data/dataset/coco_dataset.py

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved    
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.   

## engine/data/dataset/coco_eval.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.   
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools     
in the end of the file, as python3 can suppress prints with contextlib  

## engine/data/dataset/coco_utils.py

copy and modified https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
   
Copyright(c) 2023 lyuwenyu. All Rights Reserved.   

## engine/data/dataset/_dataset.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.  

## engine/data/transforms/container.py

DEIM: DETR with Improved Matching for Fast Convergence   
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)     
Copyright (c) 2024 D-FINE authors. All Rights Reserved.

## engine/data/transforms/mosaic.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

## engine/data/transforms/_transforms.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)   
Copyright(c) 2023 lyuwenyu. All Rights Reserved.  

## engine/data/transforms/__init__.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)  
Copyright(c) 2023 lyuwenyu. All Rights Reserved. 

## engine/deim/box_ops.py

# modified from torchvision to also return the union    

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

## engine/deim/deim.py

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

## engine/deim/deim_criterion.py

DEIM: DETR with Improved Matching for Fast Convergence     
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.

## engine/deim/denoising.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved. 
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     

## engine/deim/dfine_decoder.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------    
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.

## engine/deim/dfine_utils.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

## engine/deim/hybrid_encoder.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.   
---------------------------------------------------------------------------------     
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.

## engine/deim/matcher.py

Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.    

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.   

## engine/deim/postprocessor.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/deim/utils.py

DEIM: DETR with Improved Matching for Fast Convergence 
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------     
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2023 . All Rights Reserved. 

## engine/extre_module/ultralytics_nn/block.py

# Extracted from engine/extre_module/ultralytics_nn/block.py; original copyright/license retained below.

# Ultralytics YOLO 🚀, AGPL-3.0 license   

## engine/extre_module/ultralytics_nn/conv.py

# Extracted from engine/extre_module/ultralytics_nn/conv.py; original copyright/license retained below.

# Ultralytics YOLO 🚀, AGPL-3.0 license

## engine/misc/dist_utils.py

reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/misc/logger.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved 
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.

## engine/misc/profiler_utils.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved. 

## engine/misc/visualizer.py

"    
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/misc/__init__.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/optim/amp.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/optim/ema.py

D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.     

## engine/optim/lr_scheduler.py

DEIM: DETR with Improved Matching for Fast Convergence     
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.    

## engine/optim/optim.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.     

## engine/optim/warmup.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/optim/__init__.py

Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

## engine/solver/det_engine.py

DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------  
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

## engine/solver/det_solver.py

DEIM: DETR with Improved Matching for Fast Convergence 
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.     
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)   
Copyright (c) 2024 D-FINE authors. All Rights Reserved.