import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

from polyp_detector import (
    PolypDetector,
    SegmentationLoss,
    calculate_metrics,
    get_transforms
)

class Config:
    # Eğitim ayarları
    SEED = 42
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    
    # Model ayarları
    MODEL_NAME = 'polyp_segmentation'
    INPUT_SIZE = 384
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LR = 3e-4
    EPOCHS = 100
    
    # Veri yolları
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, '..', 'veri')
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    MASKS_DIR = os.path.join(DATA_DIR, 'masks')
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    VISUALIZATIONS_DIR = os.path.join(ROOT_DIR, 'visualizations')
    
    # Model kaydetme/yükleme
    LOAD_MODEL = False
    SAVE_MODEL = True
    
    # CUDA/AMP
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True
    
    # WandB entegrasyonu
    USE_WANDB = True
    WANDB_PROJECT = "polyp-segmentation"
    WANDB_ENTITY = None

class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Görüntü yükleme
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Maske yükleme
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Dönüşümler
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data_loaders():
    # Veri yollarını al
    image_paths = sorted([os.path.join(Config.IMAGES_DIR, f) for f in os.listdir(Config.IMAGES_DIR)])
    mask_paths = sorted([os.path.join(Config.MASKS_DIR, f) for f in os.listdir(Config.MASKS_DIR)])
    
    # Veriyi böl
    train_ratio = Config.TRAIN_RATIO
    valid_ratio = Config.VALID_RATIO
    
    train_size = int(len(image_paths) * train_ratio)
    valid_size = int(len(image_paths) * valid_ratio)
    
    train_image_paths = image_paths[:train_size]
    train_mask_paths = mask_paths[:train_size]
    
    valid_image_paths = image_paths[train_size:train_size+valid_size]
    valid_mask_paths = mask_paths[train_size:train_size+valid_size]
    
    test_image_paths = image_paths[train_size+valid_size:]
    test_mask_paths = mask_paths[train_size+valid_size:]
    
    # Dönüşümler
    train_transform, val_transform = get_transforms()
    
    # Dataset'ler
    train_dataset = PolypDataset(train_image_paths, train_mask_paths, train_transform)
    valid_dataset = PolypDataset(valid_image_paths, valid_mask_paths, val_transform)
    test_dataset = PolypDataset(test_image_paths, test_mask_paths, val_transform)
    
    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader

def train_fn(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    total_dice_score = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            segmentation_pred = model(images)
            loss, loss_dict = criterion(segmentation_pred, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrikleri hesapla
        metrics = calculate_metrics(segmentation_pred, masks)
        
        total_loss += loss_dict['segmentation_loss']
        total_dice_score += metrics['dice_score']
        
        pbar.set_postfix({
            'loss': loss_dict['segmentation_loss'],
            'dice': metrics['dice_score']
        })
    
    return {
        'loss': total_loss / len(loader),
        'dice_score': total_dice_score / len(loader)
    }

def eval_fn(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice_score = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            segmentation_pred = model(images)
            loss, loss_dict = criterion(segmentation_pred, masks)
            
            # Metrikleri hesapla
            metrics = calculate_metrics(segmentation_pred, masks)
            
            total_loss += loss_dict['segmentation_loss']
            total_dice_score += metrics['dice_score']
            
            pbar.set_postfix({
                'loss': loss_dict['segmentation_loss'],
                'dice': metrics['dice_score']
            })
    
    return {
        'loss': total_loss / len(loader),
        'dice_score': total_dice_score / len(loader)
    }

def visualize_predictions(model, loader, device, save_dir, num_samples=5):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            segmentation_pred = model(images)
            segmentation_pred = segmentation_pred.argmax(dim=1)
            
            for j in range(images.size(0)):
                plt.figure(figsize=(15, 5))
                
                # Orijinal görüntü
                plt.subplot(1, 3, 1)
                plt.imshow(images[j].cpu().permute(1, 2, 0))
                plt.title('Original Image')
                plt.axis('off')
                
                # Gerçek maske
                plt.subplot(1, 3, 2)
                plt.imshow(masks[j].cpu(), cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Tahmin edilen maske
                plt.subplot(1, 3, 3)
                plt.imshow(segmentation_pred[j].cpu(), cmap='gray')
                plt.title('Prediction')
                plt.axis('off')
                
                plt.savefig(os.path.join(save_dir, f'prediction_{i}_{j}.png'))
                plt.close()

def visualize_training_history(train_metrics_list, valid_metrics_list, save_dir, model_name):
    metrics = ['loss', 'dice_score']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot([m[metric] for m in train_metrics_list], label='Train')
        plt.plot([m[metric] for m in valid_metrics_list], label='Validation')
        plt.title(f'{metric.replace("_", " ").title()} Over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{model_name}_{metric}.png'))
        plt.close()

def main():
    # Seed ayarla
    seed_everything(Config.SEED)
    
    # Klasörleri oluştur
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.VISUALIZATIONS_DIR, exist_ok=True)
    
    # WandB başlat
    if Config.USE_WANDB:
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            config={
                "model_name": Config.MODEL_NAME,
                "input_size": Config.INPUT_SIZE,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LR,
                "epochs": Config.EPOCHS
            }
        )
    
    # DataLoader'ları oluştur
    train_loader, valid_loader, test_loader = create_data_loaders()
    
    # Model oluştur
    model = PolypDetector(num_segmentation_classes=2).to(Config.DEVICE)
    
    # Optimizasyon
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = SegmentationLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Model yükle
    if Config.LOAD_MODEL:
        try:
            checkpoint = torch.load(os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_best.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Eğitim döngüsü
    best_valid_loss = float('inf')
    train_metrics_list = []
    valid_metrics_list = []
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        # Eğitim
        train_metrics = train_fn(
            model, train_loader, optimizer, criterion, scaler, Config.DEVICE
        )
        
        # Validasyon
        valid_metrics = eval_fn(
            model, valid_loader, criterion, Config.DEVICE
        )
        
        # Metrikleri kaydet
        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)
        
        # WandB'ye kaydet
        if Config.USE_WANDB:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "train_dice_score": train_metrics['dice_score'],
                "valid_loss": valid_metrics['loss'],
                "valid_dice_score": valid_metrics['dice_score'],
                "epoch": epoch
            })
        
        # En iyi modeli kaydet
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics
            }
            
            if Config.SAVE_MODEL:
                torch.save(checkpoint, os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_best.pth"))
            
            # Her en iyi checkpoint'te validasyon setinden örnek tahminler oluştur
            visualize_predictions(
                model, valid_loader, Config.DEVICE,
                os.path.join(Config.VISUALIZATIONS_DIR, f'epoch_{epoch}')
            )
        
        # Son modeli kaydet
        if Config.SAVE_MODEL and (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics
            }
            torch.save(checkpoint, os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_last.pth"))
    
    # Eğitim geçmişi grafiklerini detaylı görselleştir
    visualize_training_history(
        train_metrics_list, valid_metrics_list,
        Config.VISUALIZATIONS_DIR,
        Config.MODEL_NAME
    )
    
    # Test seti üzerinde değerlendirme
    print("\nEvaluating on test set...")
    test_metrics = eval_fn(model, test_loader, criterion, Config.DEVICE)
    print(f"Test Metrics: {test_metrics}")
    
    if Config.USE_WANDB:
        wandb.log({"test_metrics": test_metrics})
        wandb.finish()

if __name__ == "__main__":
    main() 