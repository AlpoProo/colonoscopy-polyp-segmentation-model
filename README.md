# Colonoscopy Polyp Segmentation Model

## Project Overview
This project implements an AI model for segmenting polyps in colonoscopy images. The model uses a deep learning architecture to accurately identify and outline polyp regions in medical images.

## Project Structure
```
kolonoskopi/
├── modeller/
│   ├── polyp_detector.py
│   ├── train.py
│   ├── checkpoints/
│   └── visualizations/
├── veri/
│   ├── PNG/
│   │   ├── Original/
│   │   └── Ground Truth/
└── README.md
```

## Model Architecture
The model is a deep neural network that performs polyp segmentation using:
- ResNet34 encoder (pre-trained on ImageNet)
- SE (Squeeze-and-Excitation) blocks
- UNet-like decoder
- Focal Dice BCE combined loss function

### Key Features
- Strong feature extraction with ResNet34 backbone
- Feature importance learning with SE blocks
- Multi-scale feature extraction
- Better learning of difficult examples with Focal loss
- Enhanced segmentation performance with Dice loss

## Technologies Used
- PyTorch
- ResNet34
- UNet-like decoder
- SE blocks
- Albumentations (data augmentation)
- Weights & Biases (experiment tracking)

## Installation
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place colonoscopy images in `veri/PNG/Original/` directory
   - Place corresponding masks in `veri/PNG/Ground Truth/` directory
   - Ensure image and mask filenames match

## Training
To train the model:
```bash
cd modeller
python train.py
```

## Evaluation Metrics
The model is evaluated using:
- Dice Score (segmentation accuracy)
- IoU (Intersection over Union)
- Focal Dice BCE Loss

## Visualizations
During training, the following visualizations are generated:
- Loss values per epoch
- Dice Score per epoch
- IoU Score per epoch
- Model predictions and segmentation masks

## Model Outputs
For each input image, the model produces:
- Segmentation mask highlighting the polyp region

## Additional Notes
- Model uses ImageNet pre-trained ResNet34 as backbone
- SE blocks learn feature importance
- Speed improvement with mixed precision training
- Learning rate scheduling with cosine annealing with warm restarts
- Experiment tracking with Weights & Biases integration
- Comprehensive data augmentation techniques
- Automatic model checkpointing 
