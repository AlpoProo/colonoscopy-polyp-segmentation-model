import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from torchvision.models import resnet34, ResNet34_Weights
import wandb  # wandb kütüphanesini ekliyoruz

class Config:
    # Eğitim ayarları
    SEED = 42
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.2
    
    # Model ayarları
    MODEL_NAME = 'resnet_unet'
    INPUT_SIZE = 384
    BATCH_SIZE = 12
    NUM_WORKERS = 4
    LR = 3e-4
    EPOCHS = 150
    
    # Veri yolları - Proje klasörü içindeki veri dizinleri
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Bu dosyanın bulunduğu dizin
    DATA_DIR = os.path.join(ROOT_DIR, 'data')  # Proje klasörü içindeki data klasörü
    IMAGES_DIR = os.path.join(DATA_DIR, 'PNG/Original')
    MASKS_DIR = os.path.join(DATA_DIR, 'PNG/Ground Truth')
    CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
    
    # Model kaydetme/yükleme
    LOAD_MODEL = False
    SAVE_MODEL = True
    
    # CUDA/AMP
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True  # Mixed precision training
    
    # WandB entegrasyonu
    USE_WANDB = True  # WandB loglamasını aktifleştir
    WANDB_PROJECT = "polyp-segmentation"
    WANDB_ENTITY = None  # Kendi wandb kullanıcı adınızı buraya yazabilirsiniz

# Seed ayarlama
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ResNet-34 tabanlı UNet
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        # ResNet34'ü encoder olarak kullan
        if pretrained:
            self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.encoder = resnet34(weights=None)
        
        # Encoder katmanları
        self.enc1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )  # 64 kanal
        self.enc2 = nn.Sequential(
            self.encoder.maxpool,
            self.encoder.layer1
        )  # 64 kanal
        self.enc3 = self.encoder.layer2  # 128 kanal
        self.enc4 = self.encoder.layer3  # 256 kanal
        self.enc5 = self.encoder.layer4  # 512 kanal
        
        # Decoder katmanları
        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Çıkış katmanı
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # SE blokları
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(64)
        self.se3 = SEBlock(128)
        self.se4 = SEBlock(256)
        self.se5 = SEBlock(512)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [B, 64, H/2, W/2]
        enc1 = self.se1(enc1)
        
        enc2 = self.enc2(enc1)  # [B, 64, H/4, W/4]
        enc2 = self.se2(enc2)
        
        enc3 = self.enc3(enc2)  # [B, 128, H/8, W/8]
        enc3 = self.se3(enc3)
        
        enc4 = self.enc4(enc3)  # [B, 256, H/16, W/16]
        enc4 = self.se4(enc4)
        
        enc5 = self.enc5(enc4)  # [B, 512, H/32, W/32]
        enc5 = self.se5(enc5)
        
        # Decoder ile yukarı çıkma
        dec5 = self.dec5(enc5)  # [B, 256, H/16, W/16]
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))  # [B, 128, H/8, W/8]
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))  # [B, 64, H/4, W/4]
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))  # [B, 32, H/2, W/2]
        
        # Upsampling yaparak enc1 boyutuna getir
        dec1 = F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))  # [B, 32, H/2, W/2]
        
        # Son upsampling ile orijinal boyuta getir
        x_out = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False)
        x_out = self.final(x_out)  # [B, num_classes, H, W]
        
        return x_out

# Squeeze-and-Excitation Bloğu
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Custom loss: Focal Dice BCE kombinasyonu
class FocalDiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, gamma=2.0):
        super(FocalDiceBCELoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Focal BCE loss
        inputs_sigmoid = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)  # Başarı olasılığı
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce
        focal_loss = focal_loss.mean()
        
        # Dice loss
        smooth = 1.0
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        
        # Ağırlıklı kombine kayıp
        return self.weight * focal_loss + (1 - self.weight) * dice_loss

# Veri kümesi sınıfı
class PolypDataset(Dataset):
    def __init__(self, images_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = transform
        
        # Geçerli görüntü-maske çiftlerini filtrele
        self._filter_valid_pairs()
        
    def _filter_valid_pairs(self):
        """Dosyaların var olduğunu doğrula ve geçerli çiftleri tut"""
        valid_images = []
        valid_masks = []
        
        print(f"Kontrol edilen görüntü sayısı: {len(self.images_paths)}")
        
        for img_path, mask_path in zip(self.images_paths, self.masks_paths):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_images.append(img_path)
                valid_masks.append(mask_path)
            else:
                if not os.path.exists(img_path):
                    print(f"Görüntü bulunamadı: {img_path}")
                if not os.path.exists(mask_path):
                    print(f"Maske bulunamadı: {mask_path}")
                
        self.images_paths = valid_images
        self.masks_paths = valid_masks
        
        if len(self.images_paths) == 0:
            print("UYARI: Geçerli görüntü-maske çifti bulunamadı! Dosya yollarını kontrol edin.")
            print(f"Görseller dizini: {Config.IMAGES_DIR}")
            print(f"Maskeler dizini: {Config.MASKS_DIR}")
            print("Örnek bir görüntü yolu:", self.images_paths[0] if self.images_paths else "Yok")
            print("Örnek bir maske yolu:", self.masks_paths[0] if self.masks_paths else "Yok")
        else:
            print(f"Toplam {len(self.images_paths)} geçerli görüntü-maske çifti bulundu.")
        
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        
        # Görüntü yükleme
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Maske yükleme
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Maske okunamadı: {mask_path}")
        mask = mask / 255.0  # Normalize to 0-1
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension for mask
        mask = mask.unsqueeze(0)
        
        return image, mask

def get_loaders(images_paths, masks_paths):
    # Train/val split
    train_img_paths, valid_img_paths, train_mask_paths, valid_mask_paths = train_test_split(
        images_paths, masks_paths, train_size=Config.TRAIN_RATIO, random_state=Config.SEED
    )
    
    # Veri ön işleme
    train_transform = A.Compose([
        A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=20),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.2),
        A.GridDistortion(p=0.2),
        A.CLAHE(p=0.2),
        A.ColorJitter(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.ElasticTransform(p=0.2),
        A.RandomGamma(p=0.2),  # Gamma düzeltmesi
        A.OpticalDistortion(p=0.2),  # Optik distorsiyon
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    valid_transform = A.Compose([
        A.Resize(Config.INPUT_SIZE, Config.INPUT_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    # Veri kümesi oluşturma
    train_ds = PolypDataset(train_img_paths, train_mask_paths, transform=train_transform)
    valid_ds = PolypDataset(valid_img_paths, valid_mask_paths, transform=valid_transform)
    
    # DataLoader oluşturma
    train_loader = DataLoader(
        train_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, valid_loader

# Dice ve IoU metrikleri
def dice_score(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    
    return dice

def iou_score(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    
    intersection = (inputs * targets).sum()
    union = inputs.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

# Eğitim fonksiyonu
def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Mixed precision training
        with autocast(enabled=Config.USE_AMP):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Optimize step
        scaler.step(optimizer)
        scaler.update()
        
        # Metrikleri hesapla
        batch_dice = dice_score(predictions, targets)
        batch_iou = iou_score(predictions, targets)
        
        # Toplam değerleri güncelle
        total_loss += loss.item()
        total_dice += batch_dice.item()
        total_iou += batch_iou.item()
        
        # tqdm ile ilerleme çubuğunu güncelle
        loop.set_postfix(loss=loss.item(), dice=batch_dice.item(), iou=batch_iou.item())
    
    # Ortalama değerleri hesapla
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_dice, avg_iou

# Doğrulama fonksiyonu
def eval_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        loop = tqdm(loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Mixed precision
            with autocast(enabled=Config.USE_AMP):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            
            # Metrikleri hesapla
            batch_dice = dice_score(predictions, targets)
            batch_iou = iou_score(predictions, targets)
            
            # Toplam değerleri güncelle
            total_loss += loss.item()
            total_dice += batch_dice.item()
            total_iou += batch_iou.item()
            
            # tqdm ile ilerleme çubuğunu güncelle
            loop.set_postfix(loss=loss.item(), dice=batch_dice.item(), iou=batch_iou.item())
    
    # Ortalama değerleri hesapla
    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)
    
    return avg_loss, avg_dice, avg_iou

# Ana fonksiyon
def main():
    # Seed ayarla
    seed_everything(Config.SEED)
    
    # WandB başlatma
    if Config.USE_WANDB:
        wandb.init(
            project=Config.WANDB_PROJECT,
            entity=Config.WANDB_ENTITY,
            config={
                "model_name": Config.MODEL_NAME,
                "input_size": Config.INPUT_SIZE,
                "batch_size": Config.BATCH_SIZE,
                "learning_rate": Config.LR,
                "epochs": Config.EPOCHS,
                "optimizer": "Adam",
                "scheduler": "CosineAnnealingWarmRestarts",
                "loss": "FocalDiceBCE"
            }
        )
    
    # GPU Kontrol
    print(f"Using device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Dizinleri kontrol et ve oluştur
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    
    # Yol bilgilerini yazdır
    print(f"Çalışılan dizin: {os.getcwd()}")
    print(f"Kök dizin: {Config.ROOT_DIR}")
    print(f"Veri dizini: {Config.DATA_DIR}")
    print(f"Görüntü dizini: {Config.IMAGES_DIR}")
    print(f"Maske dizini: {Config.MASKS_DIR}")
    
    if not os.path.exists(Config.DATA_DIR):
        print(f"UYARI: {Config.DATA_DIR} dizini bulunamadı.")
    if not os.path.exists(Config.IMAGES_DIR):
        print(f"UYARI: {Config.IMAGES_DIR} dizini bulunamadı.")
    if not os.path.exists(Config.MASKS_DIR):
        print(f"UYARI: {Config.MASKS_DIR} dizini bulunamadı.")
    
    # PNG dosyaları kontrolü
    try:
        # Var olan dizinleri listele
        if os.path.exists(Config.DATA_DIR):
            print(f"Data dizini içeriği: {os.listdir(Config.DATA_DIR)}")
            png_dir = os.path.join(Config.DATA_DIR, 'PNG')
            if os.path.exists(png_dir):
                print(f"PNG dizini içeriği: {os.listdir(png_dir)}")
                orig_dir = os.path.join(png_dir, 'Original')
                gt_dir = os.path.join(png_dir, 'Ground Truth')
                if os.path.exists(orig_dir):
                    print(f"Original dizini içeriği: {os.listdir(orig_dir)}")
                if os.path.exists(gt_dir):
                    print(f"Ground Truth dizini içeriği: {os.listdir(gt_dir)}")
        
        # PNG dosyalarını al
        if os.path.exists(Config.IMAGES_DIR):
            all_files = os.listdir(Config.IMAGES_DIR)
            image_names = [f for f in all_files if f.lower().endswith('.png')]
            print(f"Görüntü klasöründe {len(image_names)} PNG dosyası bulundu.")
            
            if len(image_names) == 0:
                print(f"Görüntü klasöründeki tüm dosyalar: {all_files}")
                
                # Alternatif dosya uzantılarını kontrol et
                other_extensions = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                if other_extensions:
                    print(f"PNG yerine {len(other_extensions)} adet başka formatta dosya bulundu: {other_extensions[:5]}")
                    image_names = other_extensions  # Alternatif olarak diğer dosya formatlarını kullan
            
            # Dosya yollarını oluştur
            images_paths = [os.path.join(Config.IMAGES_DIR, img_name) for img_name in image_names]
            masks_paths = [os.path.join(Config.MASKS_DIR, img_name) for img_name in image_names]
            
            # İlk birkaç dosyayı kontrol et
            for img_path in images_paths[:3]:
                print(f"Görüntü var mı: {os.path.exists(img_path)} - {img_path}")
            for mask_path in masks_paths[:3]:
                print(f"Maske var mı: {os.path.exists(mask_path)} - {mask_path}")
        else:
            print(f"Görüntü dizini bulunamadı: {Config.IMAGES_DIR}")
            return
            
    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        print("Doğru dizinleri ayarladığınızdan emin olun.")
        return
    
    # Veri boş mu kontrol et
    if len(images_paths) == 0:
        print("Hata: Görüntü dosyası bulunamadı. Lütfen veri yollarınızı kontrol edin.")
        return
        
    # Veri yükleyicileri al
    train_loader, valid_loader = get_loaders(images_paths, masks_paths)

    # DataLoader'lar boş mu kontrol et
    if len(train_loader) == 0 or len(valid_loader) == 0:
        print("Hata: Veri yükleyiciler boş. Lütfen veri işleme adımlarını kontrol edin.")
        return

    # Model oluştur
    model = ResNetUNet(num_classes=1, pretrained=True).to(Config.DEVICE)
    
    # Optimizasyon
    loss_fn = FocalDiceBCELoss(weight=0.5, gamma=2.0)
    optimizer = Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    
    # Warm Restart ile kosin azalımlı öğrenme oranı
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # İlk yeniden başlatma için epoch sayısı
        T_mult=2,  # Her yeniden başlatmada T_0'ı çarpan faktör
        eta_min=1e-6
    )
    
    # FutureWarning'i önle - PyTorch versiyonuna göre GradScaler ayarla
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # Önceki kontrol noktasından devam et
    best_valid_iou = 0
    start_epoch = 0
    
    if Config.LOAD_MODEL:
        try:
            checkpoint = torch.load(os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_best.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_valid_iou = checkpoint['iou']
            print(f"Checkpoint yüklendi. Epoch {start_epoch}'dan devam ediliyor. En iyi IoU: {best_valid_iou:.4f}")
        except:
            print("Kontrol noktası yüklenemedi, sıfırdan başlıyor...")
    
    # Eğitim döngüsü
    train_losses, valid_losses = [], []
    train_dices, valid_dices = [], []
    train_ious, valid_ious = [], []
    
    for epoch in range(start_epoch, Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        # Eğitim
        train_loss, train_dice, train_iou = train_fn(train_loader, model, optimizer, loss_fn, scaler, Config.DEVICE)
        
        # Doğrulama
        valid_loss, valid_dice, valid_iou = eval_fn(valid_loader, model, loss_fn, Config.DEVICE)
        
        # Öğrenme oranını ayarla
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Geçerli öğrenme oranı: {current_lr:.8f}")
        
        # Metrik geçmişini güncelle
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_dices.append(train_dice)
        valid_dices.append(valid_dice)
        train_ious.append(train_iou)
        valid_ious.append(valid_iou)
        
        # İlerlemeyi yazdır
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Dice: {valid_dice:.4f}, Valid IoU: {valid_iou:.4f}")
        
        # WandB'ye metrikleri logla
        if Config.USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "valid_loss": valid_loss,
                "valid_dice": valid_dice,
                "valid_iou": valid_iou,
                "learning_rate": current_lr
            })
        
        # En iyi modeli kaydet
        if valid_iou > best_valid_iou and Config.SAVE_MODEL:
            print(f"IoU gelişti ({best_valid_iou:.4f} -> {valid_iou:.4f}). Model kaydediliyor...")
            best_valid_iou = valid_iou
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # load_state_dict yerine state_dict kullanmalıyız
                'loss': valid_loss,
                'dice': valid_dice,
                'iou': valid_iou,
            }
            
            torch.save(checkpoint, os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_best.pth"))
        
        # Son kontrol noktasını kaydet
        if Config.SAVE_MODEL:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # load_state_dict yerine state_dict kullanmalıyız
                'loss': valid_loss,
                'dice': valid_dice,
                'iou': valid_iou,
            }
            
            torch.save(checkpoint, os.path.join(Config.CHECKPOINTS_DIR, f"{Config.MODEL_NAME}_last.pth"))
    
    # Eğitim geçmişi grafiklerini oluştur
    plt.figure(figsize=(15, 5))
    
    # Loss grafiği
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice grafiği
    plt.subplot(1, 3, 2)
    plt.plot(train_dices, label='Train')
    plt.plot(valid_dices, label='Valid')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    # IoU grafiği
    plt.subplot(1, 3, 3)
    plt.plot(train_ious, label='Train')
    plt.plot(valid_ious, label='Valid')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_NAME}_history.png")
    
    # WandB'ye grafikleri de yükle
    if Config.USE_WANDB:
        wandb.log({"training_history": wandb.Image(plt)})
        wandb.finish()  # WandB oturumunu kapat
        
    plt.close()
    
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main() 