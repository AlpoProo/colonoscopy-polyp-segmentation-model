import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.relu(self.bn(self.conv1(x)))
        feat2 = self.relu(self.bn(self.conv2(x)))
        feat3 = self.relu(self.bn(self.conv3(x)))
        feat4 = self.relu(self.bn(self.conv4(x)))
        feat5 = self.relu(self.bn(self.conv5(self.pool(x))))
        feat5 = nn.functional.interpolate(feat5, size=size, mode='bilinear', align_corners=True)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return self.dropout(out)

class PolypDetector(nn.Module):
    def __init__(self, num_segmentation_classes=2):
        super().__init__()
        # ResNet50 encoder
        self.encoder = models.resnet50(pretrained=True)
        
        # ASPP modülü
        self.aspp = ASPP(2048, 512)
        
        # SE blokları
        self.se1 = SEBlock(2560)  # 512 * 5 = 2560 (ASPP çıktısı)
        self.se2 = SEBlock(2560)
        
        # Segmentasyon decoder'ı
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2560, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_segmentation_classes, 2, stride=2)
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        
        # ASPP ve SE blokları
        features = self.aspp(x4)
        features = self.se1(features)
        features = self.se2(features)
        
        # Segmentasyon
        segmentation = self.decoder(features)
        
        return segmentation

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, segmentation_pred, segmentation_target):
        loss = self.criterion(segmentation_pred, segmentation_target)
        return loss, {'segmentation_loss': loss.item()}

def calculate_metrics(segmentation_pred, segmentation_target):
    # Segmentasyon metrikleri (Dice score)
    segmentation_pred = segmentation_pred.argmax(dim=1)
    dice_score = torch.mean((2 * (segmentation_pred * segmentation_target).sum(dim=(2,3))) /
                          (segmentation_pred.sum(dim=(2,3)) + segmentation_target.sum(dim=(2,3)) + 1e-6))
    
    return {
        'dice_score': dice_score.item()
    }

def get_transforms():
    train_transform = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform 