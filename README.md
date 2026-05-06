# RIR-Former Evaluation

Standalone evaluation / inference script for **RIR-Former: Coordinate-Guided Transformer for Continuous Reconstruction of Room Impulse Responses**.

This repository provides a single-file evaluation pipeline that:

- Loads a pretrained RIR-Former checkpoint
- Runs inference on Room Impulse Response (RIR) `.npy` samples
- Saves ground-truth, prediction, masked-input, and mask visualizations

---

## Paper

If you use this repository in your research, please cite:

```bibtex
@INPROCEEDINGS{11462487,
  author={Xu, Shaoheng and Sun, Chunyi and Zhang, Jihui Aimee and Samarasinghe, Prasanga and Abhayapala, Thushara},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={RIR-Former: Coordinate-Guided Transformer for Continuous Reconstruction of Room Impulse Responses}, 
  year={2026},
  volume={},
  number={},
  pages={15312-15316},
  keywords={Feeds;Antennas;System-on-chip;Application specific integrated circuits;Location awareness;Mobile communication;Protocols;HTTP;LoRa;Data communication;room impulse response;RIR reconstruction;transformer models},
  doi={10.1109/ICASSP55912.2026.11462487}
}
```

Paper link: https://ieeexplore.ieee.org/document/11462487

---

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:ShaoHenry/RIR-Former.git
cd RIR-Former
```

### 2. Create a Python environment

Python 3.10 or later is recommended.

```bash
conda create -n rirformer python=3.10
conda activate rirformer
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib
```

If you are using a CUDA-enabled GPU, please install the PyTorch version that matches your CUDA version from the official PyTorch installation instructions.

---

## Download Dataset and Checkpoint

Before running evaluation, download the dataset and pretrained checkpoint from Google Drive:

https://drive.google.com/drive/folders/1XTpNppU4BbJY8WzVHkZ3dAuU-UcLZCMj?usp=sharing

After downloading, place the dataset and checkpoint in the repository directory.

A recommended directory structure is:

```text
project_root/
│
├── eval_rirformer.py
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│       ├── sample_000.npy
│       ├── sample_001.npy
│       └── ...
│
├── checkpoints/
│   └── checkpoint_main.pth
│
└── eval_images/
```

You may use different folder names, but the paths must be passed correctly to `--data_root` and `--ckpt_path` when running evaluation.

---

## Dataset Format

Each `.npy` file should contain a Python dictionary with the following fields:

```python
{
    "rir": (N_MICS, K),
    "mic_positions": (N_MICS, 3),
    "src_position": (3,)  # optional / unused during evaluation
}
```

Required keys:

- `rir`: Room impulse responses with shape `(N_MICS, K)`
- `mic_positions`: Microphone coordinates with shape `(N_MICS, 3)`

The evaluation script automatically detects `K` from the first `.npy` file in the selected split.

---

## Running Evaluation

Example command:

```bash
python eval_rirformer.py \
    --data_root dataset \
    --ckpt_path checkpoints/checkpoint_main.pth \
    --split test \
    --save_dir eval_images \
    --batch_size 8
```

In this example:

- `dataset` is the root dataset folder.
- The script reads samples from `dataset/test/` because `--split test` is used.
- `checkpoints/checkpoint_main.pth` is the pretrained checkpoint path.
- Visualization results are saved to `eval_images/`.

---

## Evaluation Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_root` | Root directory of the dataset. The script expects split folders such as `train/`, `val/`, or `test/` inside this directory. | Required |
| `--ckpt_path` | Path to the pretrained checkpoint file. | Required |
| `--split` | Dataset split to evaluate. For example, `test`, `val`, or `train`. | `test` |
| `--save_dir` | Directory where visualization images and NumPy outputs will be saved. | `eval_images` |
| `--batch_size` | Batch size for inference. | `8` |
| `--mask_ratio` | Ratio of microphones to mask during inference. | `0.7` |
| `--mask_seed` | Random seed for microphone masking. | `1234` |
| `--device` | Device used for evaluation, for example `cuda` or `cpu`. | Automatically selected |
| `--num_workers` | Number of dataloader workers. | `4` |
| `--pin_memory` | Enable pinned memory for dataloader. | Disabled |

---

## Output Visualization

All visualization results are saved to the directory specified by `--save_dir`.

For example, if you run:

```bash
python eval_rirformer.py \
    --data_root dataset \
    --ckpt_path checkpoints/checkpoint_main.pth \
    --split test \
    --save_dir eval_images
```

then all outputs will be saved under:

```text
eval_images/
```

For each sample, the script creates a separate folder:

```text
eval_images/
└── 000000_sample_name/
    ├── gt.png
    ├── pred.png
    ├── input_masked.png
    ├── mask.png
    │
    ├── gt.npy
    ├── pred.npy
    ├── input_masked.npy
    └── mask.npy
```

### Saved files

| File | Description |
|---|---|
| `gt.png` | Ground-truth RIR visualization |
| `pred.png` | Predicted RIR visualization |
| `input_masked.png` | Input RIR after microphone masking |
| `mask.png` | Binary microphone mask visualization, where observed microphones are `1` and missing microphones are `0` |
| `gt.npy` | Ground-truth RIR array |
| `pred.npy` | Predicted RIR array |
| `input_masked.npy` | Masked input RIR array |
| `mask.npy` | Binary microphone mask array |

---

## Notes

- The script performs inference only and does not compute quantitative metrics.
- Observed microphones are copied exactly from the input / ground truth during reconstruction.
- The visualization range is normalized consistently per sample across ground truth, prediction, and masked input.
- If no `.npy` files are found, check that `--data_root` and `--split` point to the correct dataset location.
- If the checkpoint cannot be loaded, check that `--ckpt_path` points to the downloaded `.pth` file.

---

## License

MIT License

Copyright (c) 2026 Shaoheng Xu and Chunyi Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
