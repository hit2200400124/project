# Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection (CVPR 2025)
**👥 Authors: Xinjie Cui, [Yuezun Li](https://yuezunli.github.io/) (corresponding author), Ao Luo, Jiaran Zhou, Junyu Dong**




![Pipeline of the proposed Forensics Adapter. ](https://github.com/OUC-VAS/ForensicsAdapter/blob/main/figures/archi.png)

---

## 📚 Resources  
| **Section**       | **Content**                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| 📄 **Paper**       | [arXiv Preprint](https://arxiv.org/abs/2411.19715)                       |
| ⚖️ **Model Weights** | [Google Drive](https://drive.google.com/file/d/1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm/view?usp=sharing) |

---

## 📊 Benchmark Comparison  
## 🖼️ Frame-Level Comparison 
🏆 **Champion Method Alert**: Our approach establishes new state-of-the-art on all frame-level benchmarks!


| Method         | Venue      | CDF-v1  | CDF-v2  | DFDC  | DFDCP  | DFD  | Avg. 📈 |
|----------------|------------|---------|---------|-------|--------|------|-------|
| SPSL       | CVPR'21    | 0.815   | 0.765   | 0.704 | 0.741  | 0.812| 0.767 |
| SRM       | CVPR'21    | 0.793   | 0.755   | 0.700 | 0.741  | 0.812| 0.760 |
| Reece       | CVPR'22    | 0.768   | 0.732   | 0.713 | 0.734  | 0.812| 0.752 |
| SBI        | CVPR'22    | -       | 0.813   | -     | 0.799  | 0.774| -     |
| UCF        | ICCV'23    | 0.779   | 0.753   | 0.719 | 0.759  | 0.807| 0.763 |
| ED          | AAAI'24    | 0.818   | 0.864   | 0.721 | 0.851  | -    | -     |
| LSDA      | CVPR'24    | 0.867   | 0.830   | 0.736 | 0.815  | 0.880| 0.826 |
| CFM        | TIFS'24    | -       | 0.828   | -     | 0.758  | 0.915| -     |
| **Ours**       | **CVPR'25**      | 🥇 **0.914** | 🥇 **0.900** | 🥇 **0.843** | 🥇 **0.890** | 🥇 **0.933** | 🥇 **0.896** |

## 🎥 Video-Level Comparison

🏆 **Champion Method Alert**: Our approach achieves new SOTA performance across all video-level benchmarks!

| Method             | Venue      | CDF-v2  | DFDC  | DFDCP  |
|--------------------|------------|---------|-------|--------|
| TALL          | ICCV'23    | 0.908   | 0.768 | -      |
| AltFreezing    | CVPR'23    | 0.895   | -     | -      |
| SeeABLE       | ICCV'23    | 0.873   | 0.759 | 0.863  |
| IID           | CVPR'23    | 0.838   | -     | 0.812  |
| TALL++        | IJCV'24    | 0.920   | 0.785 | -      |
| SAM            | CVPR'24    | 0.890   | -     | -      |
| SBI           | CVPR'22    | 0.932   | 0.724 | 0.862  |
| CADDM       | CVPR'23    | 0.939   | 0.739 | -      |
| SFDG         | CVPR'23    | 0.758   | 0.736 | -      |
| LAA-NET       | CVPR'24    | 0.954   | -     | 0.869  |
| LSDA          | CVPR'24    | 0.911   | 0.770 | -      |
| CFM        | TIFS'24    | 0.897   | -     | 0.802  |
| **Ours**           | **CVPR'25**      | 🥇 **0.957** | 🥇 **0.872** | 🥇 **0.929** |

---


## 🚀 Start

- [⏳ Environment Setup](#-environment-setup)
- [📂 Dataset](#-dataset)
- [🏋️ Training](#-training)
- [🧪 Testing](#-testing)
- [📝 Citation](#-citation)


## ⏳ Environment Setup
Ensure your environment meets the following requirements:

- 🐍 Python 3.9
- 🔥 PyTorch 1.11
- 🚀 CUDA 11.3

Install dependencies:

```bash
conda create -n FA python=3.9
conda activate FA
sh install.sh
```

## 📂 Dataset

We use multiple datasets for training and evaluation:

- FF++
- DFDC
- DFDCP
- DFD
- CD1/CD2

The dataset downloading and processing procedures can be referred to the implementation provided in [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) .


## 🏋️ Training
From the ForensicsAdapter directory, set `dataset_json_folder`, `log_dir`, and `device` in `config/train.yaml`.
Use the paths in your DeepfakeBench JSON to point to the aligned RGB frames and masks. Set
`clip_download_root` to your CLIP cache or `clip_model_name` to a local OpenAI CLIP checkpoint.
The supplied training configuration uses the FF++ c23 train split and selects weights on the FF++ validation split.

```bash
python train.py --config_path config/train.yaml
```

## 🧪 Testing
Set the dataset paths and device in `config/test.yaml`. Use the `frequency_fusion`,
`conv_moe_mode`, and `conv_moe_top_k` values from the training run.

```bash
python test.py --detector_path config/test.yaml --weights_path /path/to/ckpt_best.pth
```

## Frequency fusion experiments

This extension to the [original Forensics Adapter](https://github.com/OUC-VAS/ForensicsAdapter)
places a frequency residual after Adapter block 3 and its CLIP feature fusion. It extracts
mid- and high-band log energies from pre-normalization luminance with a 16-by-16 DCT per Adapter
patch. The coverage head is supervised by mask area fractions and trains from detached visual tokens.
At inference the reference selection uses predicted coverage.

| `frequency_fusion` | Input to the frequency projection |
| --- | --- |
| `off` | Adapter and CLIP features only |
| `absolute` | Per-patch mid- and high-band log energy |
| `uniform` | Absolute energy and signed difference from the neighborhood mean |
| `guided` | Absolute energy and signed difference from the low-coverage neighbor mean |

Both relative modes fall back to absolute energy when fewer than three neighbors in the 5-by-5
window have predicted coverage at most 0.2. Evaluation reports the proportion of patches with an
active reference, including separate rates for real and fake frames.
Train each mode from scratch for controlled comparisons. The CLIP feature projections are registered
in this version, so older detector checkpoints lack those learned projection weights.

## Convolutional MoE experiments

| `conv_moe_mode` | Placement |
| --- | --- |
| `off` | Uses the existing Adapter and supports checkpoints trained without MoE. |
| `post_fusion` | Adds the MoE residual after block 3 and its CLIP fusion. DCT and MoE both read the same fused visual tokens when enabled together. |
| `ffn` | Adds the MoE residual to visual tokens alongside the block 4 MLP after self-attention. |

Both MoE placements use a shared convolution and route each image to CDC, ADC, or RDC. The
default `conv_moe_top_k: 1` selects one expert with its probability from the full router softmax.
Set `conv_moe_top_k: 2` to select two experts and normalize their weights over the selected pair.
`conv_moe_balance_weight` controls the routing loss during training. Evaluation reports each
expert's selection rate and mean weight across frames.

For the placement comparison, train `off`, `post_fusion`, and `ffn` runs with
`frequency_fusion: 'off'` and matching seeds and training schedules. To train the joint model,
set `conv_moe_mode: 'ffn'` and `frequency_fusion: 'guided'`.

## 📝 Citation

If our work is useful for your research, please cite it as follows:

```bibtex
@InProceedings{Cui_2025_CVPR,
    author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
    title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {19207-19217}
}
```
## License
This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?utm_source=chatgpt.com)).
This repository is intended for academic research use only. Commercial use is prohibited.
Please refer to the LICENSE file for more details.


