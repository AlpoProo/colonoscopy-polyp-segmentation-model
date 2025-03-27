# Colonoscopy Polyp Detection and Classification Model

## Project Overview
This project implements an AI model for detecting and classifying polyps in colonoscopy images. The model performs both binary classification (polyp/no polyp) and multi-class classification of polyp types, along with polyp segmentation.

## Project Structure
```
kolonoskopi/
├── modeller/
│   ├── polyp_detector.py
│   ├── train.py
│   ├── checkpoints/
│   └── visualizations/
├── veri/
│   ├── images/
│   └── masks/
└── README.md
```

## Model Architecture
The model is a multi-task neural network that performs:
- Polyp detection (binary classification)
- Polyp type classification (multi-class)
- Polyp segmentation

Key components:
- ResNet50 encoder (pre-trained on ImageNet)
- ASPP (Atrous Spatial Pyramid Pooling) for multi-scale feature extraction
- SE (Squeeze-and-Excitation) blocks for feature importance learning
- UNet-like decoder for segmentation

## Technologies Used
- PyTorch
- ResNet50
- UNet-like decoder
- ASPP
- SE blocks

## Installation
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place colonoscopy images in `veri/images/`
   - Place corresponding masks in `veri/masks/`
   - Ensure image and mask filenames match

## Training
To train the model:
```bash
cd modeller
python train.py
```

## Evaluation Metrics
The model is evaluated using:
- F1 Score for polyp detection
- Accuracy for polyp type classification
- Dice Score for segmentation

## Visualizations
During training, the following visualizations are generated:
- Loss values over epochs
- Accuracy scores
- Model predictions with segmentation masks

## Model Outputs
For each input image, the model produces:
1. Polyp detection (present/absent)
2. Polyp type classification
3. Segmentation mask highlighting the polyp region

## Additional Notes
- The model uses ImageNet pre-trained ResNet50 as the backbone
- ASPP module captures multi-scale contextual information
- SE blocks learn feature importance
- Mixed precision training is used for improved speed
- Cosine annealing with warm restarts for learning rate scheduling
- WandB integration for experiment tracking
