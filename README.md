# Multimodal Flood Mapping with Deep Learning (Sentinel-2 + DEM)

## Overview

This project develops a deep learning framework for segmenting water bodies from satellite imagery using Sentinel-2 multispectral data.

The objective is to evaluate whether combining spectral, topographic, and engineered features improves flood detection performance and generalization across unseen geographic regions.

We compare multiple feature configurations:
- Raw Sentinel-2 bands
- Sentinel-2 + DEM
- Spectral indices + DEM
- Spectral indices + HSV + DEM

Two architectures are evaluated:
- U-Net
- MA-Net

The focus is on understanding the trade-off between feature complexity and generalization.

## Key Findings

- Multimodal inputs (DEM + indices + HSV) achieve the highest validation performance
- However, they do not generalize well to unseen test scenes
- The best overall performance is achieved using **raw Sentinel-2 data only**
- Additional features can introduce overfitting rather than improving robustness

This highlights that **simpler models can outperform complex multimodal systems in real-world flood mapping scenarios**

## Dataset

- Source: Sentinel-2 multispectral imagery + DEM
- Patch size: 256 × 256
- Split:
  - Train: 2 scenes
  - Validation: 2 scenes
  - Test: 10 scenes

Total patches:
- Train: ~1200
- Validation: ~520
- Test: ~834

Labels are binary water masks.

## Models

Two architectures were implemented using `segmentation_models_pytorch`:

- U-Net (baseline)
- MA-Net (attention-based)

Encoder:
- MobileNetV2 (random initialization)

Loss function:
- Focal Loss (to handle class imbalance)

## Results

| Model | Feature Set | Test F1 | Test IoU |
|------|------------|--------|---------|
| U-Net | raw_s2 | **0.872** | **0.773** |
| U-Net | raw_idx_hsv_dem | 0.708 | 0.549 |
| MA-Net | raw_idx_hsv_dem | 0.703 | 0.542 |

Key insight:
- Multimodal features improve validation performance but degrade generalization

## Project Structure
flood-segmentation/
│
├── FRS 2026. Floods Monitoring/
│   ├── dem
│   ├── masks
│   ├── multispectral 
│   ├── osm
│   ├── tips.txt
│
├── Scripts/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing_and_dataset.ipynb
│   ├── 03_models_focal_fast copy.ipynb
│   ├── 03_models_focal_fast.ipynb
│   ├── 03_models_screening_first.ipynb
│   ├── 03_models_screening_first copy.ipynb

│
├── step2_artifacts/
│   ├── step2_bundle.json
│   ├── train_index.pkl
│   ├── val_index.pkl
│   ├── test_index.pkl
│
├── Results/
│   ├── figures
│   ├── csv files
│
├── README.md
└── requirements.txt

## How to Run

1. Install dependencies:
pip install -r requirements.txt
2. Complete dataset analysis:
- 01_dataset_analysis.ipynb

3. Run Preprocessing:
- 02_preprocessing_and_dataset.ipynb

4. Train models:
- 03_models_focal_fast copy.ipynb
- 03_models_focal_fast.ipynb
- 03_models_screening_first.ipynb
- 03_models_screening_first copy.ipynb

## Requirements

- Python 3.9+
- PyTorch
- segmentation_models_pytorch
- numpy
- matplotlib
