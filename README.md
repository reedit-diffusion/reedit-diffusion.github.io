# ReEdit: Multimodal Exemplar-Based Image Editing with Diffusion Models (WACV 2025)

## [<a href="https://reedit-diffusion.github.io/" target="_blank">Project Page</a>]
[![arXiv](https://img.shields.io/badge/arXiv-ReEdit-b31b1b.svg)](https://arxiv.org/abs/2411.03982)
[![GitHub](https://img.shields.io/badge/GitHub-ReEdit-4CAF50)](https://github.com/reedit-diffusion/reedit-diffusion.github.io)

![ReEdit Overview](static/images/banner.jpg)

ReEdit is an efficient end-to-end optimization-free framework for exemplar-based image editing. Unlike existing approaches, it doesn't require fine-tuning or optimization during inference time.

## Overview

Given a pair of exemplar images (original and edited), ReEdit captures the edit and applies it to a test image to obtain the corresponding edited version. The framework consists of three main components:

1. **Image Space Edit Capture**: Uses pretrained adapter modules to capture edits in the image embedding space
2. **Text Space Edit Capture**: Incorporates multimodal VLMs for detailed reasoning and edit description
3. **Content Preservation**: Conditions image generation on test image features and self-attention maps

## Key Features

- No fine-tuning or optimization required during inference
- ~4x faster than baseline methods
- Preserves original image structure while applying edits
- Works with various types of edits
- Model-agnostic (independent of base diffusion model)

## Installation

This project has 2 different conda environments `llava` and `reedit`. You can set up these environments manually by running:

### llava environment
```bash
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava 
pip install --upgrade pip
pip install -e .
pip install protobuf
```

### reedit environment
```bash
conda create -n reedit python=3.9 -y
conda activate reedit
pip install -r requirements.txt
```

## Usage
First add your exemplar pairs in the `data` directory in the following format:
```
data
└── add_butterfly
    ├── 0_0.png
    ├── 0_1.png
    └── 1_0.png
```

### Generate LLaVA captions
To generate LLaVA captions for your exemplar pairs, run:
```bash
python preprocess-llava.py --directory data
cd LLaVA
python edit.py --img_fol ../data --res_fol ../llava_results
python get_caption.py --img_fol ../data --res_fol ../llava_results
python3 truncate_caption.py --res_fol ../llava_results
```

### Preprocessing for PNP
```bash
python3 preprocess.py --data_path data
```

### Run the altered PNP script
```bash
python3 preprocess.py --data_path data
python3 pnp.py --name reedit --group reedit
```

## Dataset

The project includes a curated dataset of 1474 exemplar pairs covering various edit types:

- Global Style Transfer (428 pairs)
- Background Change (212 pairs)
- Localized Style Transfer (290 pairs)
- Object Replacement (366 pairs)
- Motion Edit (14 pairs)
- Object Insertion (164 pairs)

## Architecture

ReEdit combines several key components:

1. **IP-Adapter**: Handles image prompt conditioning
2. **LLaVA Integration**: Provides detailed reasoning and text descriptions
3. **PNP Module**: Maintains the structure of the test image while performing the edit

## Performance

Compared to baselines:
- 4x faster inference time
- Better consistency in non-edited regions
- Higher edit accuracy
- Improved structure preservation

## Citation

```
@article{srivastava2024reedit,
  title={ReEdit: Multimodal Exemplar-Based Image Editing with Diffusion Models},
  author={Srivastava, Ashutosh and Menta, Tarun Ram and Java, Abhinav and Jadhav, Avadhoot and Singh, Silky and Jandial, Surgan and Krishnamurthy, Balaji},
  journal={arXiv preprint arXiv:2411.03982},
  year={2024}
}
```
