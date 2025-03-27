# Colonoscopy Polyp Segmentation Model

## Project Overview
This project implements an AI model for segmenting polyps in colonoscopy images. The model uses a deep learning architecture to accurately identify and outline polyp regions in medical images.

## Project Structure
```
colonoscopy/
├── mdoels/
│   ├── polyp_detector.py
│   ├── train.py
│   ├── checkpoints/
│   └── visualizations/
├── data/
│   ├── images/
│   └── masks/
└── README.md
```

## Model Architecture
The model is a deep neural network that performs polyp segmentation using:
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
cd models
python train.py
```

## Evaluation Metrics
The model is evaluated using:
- Dice Score for segmentation accuracy

## Visualizations
During training, the following visualizations are generated:
- Loss values over epochs
- Dice Score over epochs
- Model predictions with segmentation masks

## Model Outputs
For each input image, the model produces:
- Segmentation mask highlighting the polyp region

## Additional Notes
- The model uses ImageNet pre-trained ResNet50 as the backbone
- ASPP module captures multi-scale contextual information
- SE blocks learn feature importance
- Mixed precision training is used for improved speed
- Cosine annealing with warm restarts for learning rate scheduling
- WandB integration for experiment tracking 
