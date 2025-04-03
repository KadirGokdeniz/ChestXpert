# models/backbones/alexnet.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights

class AlexNetBackbone(nn.Module):
    """
    AlexNet tabanlı öznitelik çıkarıcı omurga.
    """
    
    def __init__(self, pretrained=True, freeze_layers=True):
        """
        Args:
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
            freeze_layers: İlk katmanları dondur (transfer öğrenme için)
        """
        super(AlexNetBackbone, self).__init__()
        
        # AlexNet modelini yükle
        if pretrained:
            self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.model = models.alexnet(weights=None)
        
        # AlexNet'in konvolüsyon katmanlarını ayır
        self.features = self.model.features
        
        # Ortalama havuzlama ve aralama katmanı
        self.avgpool = self.model.avgpool
        
        # Sınıflandırıcının son katmanını kaldır (öznitelik çıkarıcı olarak kullanmak için)
        self.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        # Özellik boyutunu kaydet
        self.feature_dim = 4096  # AlexNet'in son katmandan önceki boyutu
        
        # İlk katmanları dondur (isteğe bağlı)
        if freeze_layers:
            # AlexNet'in ilk konvolüsyon katmanlarını dondur
            for i, param in enumerate(self.features.parameters()):
                # İlk 3 konvolüsyon bloğu
                if i < 6:  # ilk 3 konvolüsyon katmanının parametreleri
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            features: Çıktı öznitelik tensörü [batch_size, feature_dim]
        """
        # Konvolüsyon katmanları
        x = self.features(x)
        
        # Ortalama havuzlama
        x = self.avgpool(x)
        
        # Düzleştir
        x = torch.flatten(x, 1)
        
        # Sınıflandırıcı katmanları (son katman hariç)
        x = self.classifier(x)
        
        return x
    
    def extract_feature_maps(self, x):
        """
        AlexNet'in farklı katmanlarından öznitelik haritalarını çıkarır.
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            feature_maps: Farklı katmanlardan öznitelik haritaları sözlüğü
        """
        feature_maps = {}
        
        # AlexNet'in konvolüsyon katmanlarını ayrıştır
        # AlexNet 5 konvolüsyon katmanına sahiptir, bunlar features içinde sıralıdır
        
        # Katman 1: Konvolüsyon + ReLU + MaxPool
        x = self.features[0](x)  # Conv2d
        feature_maps['conv1'] = x
        
        x = self.features[1](x)  # ReLU
        x = self.features[2](x)  # MaxPool
        feature_maps['pool1'] = x
        
        # Katman 2: Konvolüsyon + ReLU + MaxPool
        x = self.features[3](x)  # Conv2d
        feature_maps['conv2'] = x
        
        x = self.features[4](x)  # ReLU
        x = self.features[5](x)  # MaxPool
        feature_maps['pool2'] = x
        
        # Katman 3: Konvolüsyon + ReLU
        x = self.features[6](x)  # Conv2d
        feature_maps['conv3'] = x
        
        x = self.features[7](x)  # ReLU
        
        # Katman 4: Konvolüsyon + ReLU
        x = self.features[8](x)  # Conv2d
        feature_maps['conv4'] = x
        
        x = self.features[9](x)  # ReLU
        
        # Katman 5: Konvolüsyon + ReLU + MaxPool
        x = self.features[10](x)  # Conv2d
        feature_maps['conv5'] = x
        
        x = self.features[11](x)  # ReLU
        x = self.features[12](x)  # MaxPool
        feature_maps['pool5'] = x
        
        # Adaptif ortalama havuzlama
        x = self.avgpool(x)
        feature_maps['avgpool'] = x
        
        # Düzleştir
        x = torch.flatten(x, 1)
        feature_maps['flatten'] = x
        
        # Tamamen bağlı katmanlar
        x = self.classifier[0](x)  # Dropout
        x = self.classifier[1](x)  # Linear
        x = self.classifier[2](x)  # ReLU
        feature_maps['fc6'] = x
        
        x = self.classifier[3](x)  # Dropout
        x = self.classifier[4](x)  # Linear
        x = self.classifier[5](x)  # ReLU
        feature_maps['fc7'] = x
        
        return feature_maps

# Test için
if __name__ == "__main__":
    # Yapay girdi tensörü oluştur
    x = torch.randn(4, 3, 224, 224)
    
    # Modeli test et
    model = AlexNetBackbone(pretrained=True)
    print(f"Model yapısı: {model}")
    
    # Forward pass ile çıktı boyutunu kontrol et
    out = model(x)
    print(f"Çıktı tensör boyutu: {out.shape}")
    print(f"Öznitelik boyutu: {model.feature_dim}")
    
    # Eğitilebilir parametre sayısını hesapla
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    print(f"Donmuş parametre oranı: {(total_params - trainable_params) / total_params:.2%}")
    
    # AlexNet'in öznitelik haritalarını incele
    feature_maps = model.extract_feature_maps(x)
    for name, fm in feature_maps.items():
        print(f"{name}: {fm.shape}")