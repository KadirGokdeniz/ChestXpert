# models/backbones/resnet.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights

class ResNetBackbone(nn.Module):
    """
    ResNet tabanlı öznitelik çıkarıcı omurga.
    ResNet18, ResNet50 veya ResNet101 kullanımını destekler.
    """
    
    def __init__(self, variant='resnet50', pretrained=True, freeze_layers=True):
        """
        Args:
            variant: ResNet varyantı ('resnet18', 'resnet50', 'resnet101')
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
            freeze_layers: İlk katmanları dondur (transfer öğrenme için)
        """
        super(ResNetBackbone, self).__init__()
        
        # Model varyantını seç
        if variant == 'resnet18':
            if pretrained:
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
        elif variant == 'resnet50':
            if pretrained:
                self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.model = models.resnet50(weights=None)
        elif variant == 'resnet101':
            if pretrained:
                self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            else:
                self.model = models.resnet101(weights=None)
        else:
            raise ValueError(f"Desteklenmeyen ResNet varyantı: {variant}")
        
        # Özellik boyutunu kaydet
        self.feature_dim = self.model.fc.in_features
        
        # Son sınıflandırıcıyı kaldır (öznitelik çıkarıcı olarak kullanmak için)
        self.model.fc = nn.Identity()
        
        # BBoxFusionModel ile uyumluluk için features özniteliğini ekle
        # Bu, diğer CNN mimarileriyle (VGG, AlexNet, vb.) uyumluluğu sağlar
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        )
        
        # İlk katmanları dondur (isteğe bağlı)
        if freeze_layers:
            # ResNet'in ilk katmanlarını dondur
            layers_to_freeze = [
                self.model.conv1,
                self.model.bn1,
                self.model.layer1,
                self.model.layer2
            ]
            
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            features: Çıktı öznitelik tensörü [batch_size, feature_dim]
        """
        # Doğrudan model ile forward pass yapabiliriz
        # Model.fc nn.Identity olduğu için öznitelik vektörü döndürür
        return self.model(x)
    
    def extract_feature_maps(self, x):
        """
        ResNet'in farklı katmanlarından öznitelik haritalarını çıkarır
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            feature_maps: Farklı katmanlardan öznitelik haritaları sözlüğü
        """
        feature_maps = {}
        
        # İlk katmanlar
        x = self.model.conv1(x)
        feature_maps['conv1'] = x
        
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        feature_maps['maxpool'] = x
        
        # ResNet blokları
        x = self.model.layer1(x)
        feature_maps['layer1'] = x
        
        x = self.model.layer2(x)
        feature_maps['layer2'] = x
        
        x = self.model.layer3(x)
        feature_maps['layer3'] = x
        
        x = self.model.layer4(x)
        feature_maps['layer4'] = x
        
        # Global average pooling
        x = self.model.avgpool(x)
        feature_maps['avgpool'] = x
        
        # Düzleştir
        x = torch.flatten(x, 1)
        feature_maps['flatten'] = x
        
        return feature_maps

# Test için
if __name__ == "__main__":
    # Yapay girdi tensörü oluştur
    x = torch.randn(4, 3, 224, 224)
    
    # Modeli test et
    model = ResNetBackbone(variant='resnet50', pretrained=True)
    print(f"Model yapısı: {model}")
    
    # Forward pass ile çıktı boyutunu kontrol et
    out = model(x)
    print(f"Çıktı tensör boyutu: {out.shape}")
    print(f"Öznitelik boyutu: {model.feature_dim}")
    
    # Features katmanını kontrol et
    features_out = model.features(x)
    print(f"Features çıktı boyutu: {features_out.shape}")
    
    # Eğitilebilir parametre sayısını hesapla
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    print(f"Donmuş parametre oranı: {(total_params - trainable_params) / total_params:.2%}")