# Learning to Generate Gradients for Test-Time Adaptation via Test-Time Training Layers

This is the official project repository for Learning to Generate Gradients for Test-Time Adaptation via Test-Time Training Layers by Qi Deng, Shuaicheng Niu, Ronghao Zhang Yaofo Chen Runhao Zeng Jian Chen Xiping Hu(AAAI 2025)
<!-- 
* 1️⃣ FOA conducts model learning at test time to adapt a pre-trained model to test data that has distributional shifts ☀️ 🌧 ❄️, such as corruptions, simulation-to-real discrepancies, and other differences between training and testing data.
* 2️⃣ FOA performs adaptation on both _input_ and _output_ levels, which avoids modification to model parameters and adapts in a backpropagation-free manner. Consequently, FOA offers the following benefits:
  1) reduces memory usage significantly, _e.g._, 5,165MB (Tent) $\rightarrow$ 832MB when using ViTBase and a _BS_ of 64. 
  2) compatible with quantized models, which typically do not support backpropagation.
  3) compatible with models on specialized chips, where parameters are hardcoded and non-modifiable. 

<p align="center">
<img src="figures/foa.png" alt="FOA" width="100%" align=center />
</p> -->

introduction of MGTTA ...

**Dependencies Installation:**
```
conda create -n FOA python=3.8.18
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.9.10
pip install transformers
```

<!-- pip install cma 
pip install einops
pip install scipy -->

**Data Preparation:**

This repository contains code for evaluation on ImageNet-C/R/Sketch/A with VitBase. But feel free to use your own data and models! Please check [here 🔗](dataset/README.md) for a detailed guide on preparing these datasets.

**Usage**
```python
import tta_library.mgtta as mgtta
from models.vpt import FOAViT

model = TODO_model()
model = FOAViT(model)
mgg = LOAD_trained_mgg()

adapt_model = mgtta.MGTTA(model, mgg, args.adapt_lr)
train_loader = TODO_loader()
adapt_model.obtain_origin_stat(train_loader)
adapt_model.configure_model()

outputs = adapt_model(inputs)
```

# Example: ImageNet-C Experiments

**Usage:**
...

<!-- ```
python3 main.py \
    --data path/to/imagenet \
    --data_v2 path/to/imagenet-v2 \
    --data_sketch path/to/imagenet-sketch \
    --data_corruption path/to/imagenet-c \
    --data_rendition path/to/imagenet-r \
    --algorithm [tent/foa/lame/t3a/sar/cotta] \ 
``` -->

<!-- 
**Experimental Results**

The Table below demonstrates the result of both full precision and quantized ViTs. The reported average accuracy (\%, $\uparrow$) / ECE (\%, $\downarrow$) is averaged over 15 different corruption types in ImageNet-C (severity level 5).

|         | ViT (full precision, 32-bit) | ViT (8-bit) | ViT (6-bit) |
| ------- | :----------: | :---------: | :---------: |
| NoAdapt | 55.5 / 10.5  | 54.1 / 10.8 | 47.7 / 9.9  |
| LAME    | 54.1 / 11.0  | 52.5 / 12.4 | 45.8 / 10.4 |
| T3A     | 56.9 / 26.8  | 55.1 / 25.9 | 45.4 / 30.1 |
| Tent    | 59.6 / 18.5  |      -      |      -      |
| CoTTA   |  61.7 / 6.5  |      -      |      -      |
| SAR     |  62.7 / 7.0  |      -      |      -      |
| FOA     |  66.3 / 3.2  | 63.5 / 3.8  | 55.8 / 5.5  |

Please see our [PAPER 🔗]() for more detailed results. -->

# Correspondence

Please contact Qi Deng by[dengqi.kei at gmail.com] and Shuaicheng Niu by [shuaicheng.niu at ntu.edu.sg] and Guohao Chen by [secasper at mail.scut.edu.cn] and Ronghaoa Zhang by [zhangronghao16 at gmail.com] if you have any questions. 📬

# Citation

If our MGTTA method or paper is helpful in your research, please consider citing our paper:

```

@inproceedings{deng2025learning,
  title={Learning to Generate Gradients for Test-Time Adaptation via Test-Time Training Layers},
  author={Deng, Qi and Niu, Shuaicheng and Zhang, Ronghao and Chen, Yaofo and Zeng, Runhao and Chen, Jian and Hu, Xiping},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

# Acknowledgment

The code is inspired by [FOA 🔗](https://github.com/mr-eggplant/FOA) and [TTT 🔗](https://github.com/test-time-training/ttt-lm-pytorch).