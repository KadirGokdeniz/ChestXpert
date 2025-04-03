# models/backbones/mobilenet.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

class MobileNetBackbone(nn.Module):
    """
    MobileNet tabanlı öznitelik çıkarıcı omurga.
    MobileNetV2, MobileNetV3-Small veya MobileNetV3-Large kullanımını destekler.
    """
    
    def __init__(self, variant='mobilenet_v2', pretrained=True, freeze_layers=True):
        """
        Args:
            variant: MobileNet varyantı ('mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large')
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
            freeze_layers: İlk katmanları dondur (transfer öğrenme için)
        """
        super(MobileNetBackbone, self).__init__()
        
        # Model varyantını seç
        if variant == 'mobilenet_v2':
            if pretrained:
                self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                self.model = models.mobilenet_v2(weights=None)
            
            # Öznitelik boyutunu kaydet
            self.feature_dim = self.model.classifier[1].in_features  # 1280
            
            # Son sınıflandırıcıyı kaldır
            self.model.classifier = nn.Identity()
            
        elif variant == 'mobilenet_v3_small':
            if pretrained:
                self.model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                self.model = models.mobilenet_v3_small(weights=None)
            
            # Öznitelik boyutunu kaydet
            self.feature_dim = self.model.classifier[3].in_features  # 1024
            
            # Son sınıflandırıcıyı kaldır
            self.model.classifier = nn.Identity()
            
        elif variant == 'mobilenet_v3_large':
            if pretrained:
                self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            else:
                self.model = models.mobilenet_v3_large(weights=None)
            
            # Öznitelik boyutunu kaydet
            self.feature_dim = self.model.classifier[3].in_features  # 1280
            
            # Son sınıflandırıcıyı kaldır
            self.model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Desteklenmeyen MobileNet varyantı: {variant}")
        
        # Model varyantını kaydet
        self.variant = variant
        
        # İlk katmanları dondur (isteğe bağlı)
        if freeze_layers:
            # İlk blokları dondur
            if variant == 'mobilenet_v2':
                # V2 için genellikle ilk 3 blok (features[0:5]) dondurulabilir
                for i in range(5):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = False
            else:
                # V3 için genellikle ilk 3 blok dondurulabilir
                for i in range(3):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            features: Çıktı öznitelik tensörü [batch_size, feature_dim]
        """
        # MobileNetV2 için
        if self.variant == 'mobilenet_v2':
            x = self.model.features(x)
            # Global ortalamalı havuzlama
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            # Düzleştir
            x = torch.flatten(x, 1)
            return x
        
        # MobileNetV3 için
        else:
            # Tüm features katmanlarını uygula
            x = self.model.features(x)
            # Global ortalamalı havuzlama
            x = self.model.avgpool(x)
            # Düzleştir
            x = torch.flatten(x, 1)
            return x
    
    def extract_feature_maps(self, x):
        """
        Farklı katmanlardan öznitelik haritalarını çıkarır. 
        Her MobileNet sürümü için katman sayısı ve yapısı değişir.
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            feature_maps: Farklı katmanlardan öznitelik haritaları sözlüğü
        """
        feature_maps = {}
        
        # MobileNetV2 features katmanlarından öznitelik haritalarını çıkar
        if self.variant == 'mobilenet_v2':
            current_tensor = x
            
            # Her bir konvolüsyon/blok için öznitelik haritasını kaydet
            for i, layer in enumerate(self.model.features):
                current_tensor = layer(current_tensor)
                feature_maps[f'block_{i}'] = current_tensor
            
            # Global ortalamalı havuzlama
            pooled = nn.functional.adaptive_avg_pool2d(current_tensor, (1, 1))
            feature_maps['global_pool'] = pooled
            
            # Düzleştir
            flattened = torch.flatten(pooled, 1)
            feature_maps['flatten'] = flattened
            
        # MobileNetV3 için
        else:
            current_tensor = x
            
            # Her bir konvolüsyon/blok için öznitelik haritasını kaydet
            for i, layer in enumerate(self.model.features):
                current_tensor = layer(current_tensor)
                feature_maps[f'block_{i}'] = current_tensor
            
            # Global ortalamalı havuzlama
            pooled = self.model.avgpool(current_tensor)
            feature_maps['global_pool'] = pooled
            
            # Düzleştir
            flattened = torch.flatten(pooled, 1)
            feature_maps['flatten'] = flattened
        
        return feature_maps

# Test için
if __name__ == "__main__":
    # Yapay girdi tensörü oluştur
    x = torch.randn(4, 3, 224, 224)
    
    # Modeli test et
    model = MobileNetBackbone(variant='mobilenet_v2', pretrained=True)
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
    
    # Farklı modellerle karşılaştırma
    models_to_compare = [
        ('mobilenet_v2', MobileNetBackbone(variant='mobilenet_v2', pretrained=True)),
        ('mobilenet_v3_small', MobileNetBackbone(variant='mobilenet_v3_small', pretrained=True)),
        ('mobilenet_v3_large', MobileNetBackbone(variant='mobilenet_v3_large', pretrained=True))
    ]
    
    print("\nMobileNet Varyantları Karşılaştırması:")
    for name, model in models_to_compare:
        out = model(x)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: Öznitelik boyutu = {model.feature_dim}, Parametre sayısı = {total_params:,}")