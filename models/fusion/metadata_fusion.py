import torch
import torch.nn as nn
import torch.nn.functional as F

class MetadataFusionModel(nn.Module):
    """
    Metadata entegrasyonu için model sınıfı.
    Görüntü öznitelikleri ve metadata özniteliklerini birleştirir.
    """
    
    def __init__(self, backbone, num_classes, metadata_dim=3, fusion_method='concat'):
        """
        Args:
            backbone: Görüntü öznitelik çıkarıcı
            num_classes: Sınıf sayısı
            metadata_dim: Metadata öznitelik boyutu (yaş, cinsiyet, görüntü pozisyonu vb.)
            fusion_method: Füzyon yöntemi ('concat', 'attention', 'gating', 'film')
        """
        super(MetadataFusionModel, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.fusion_method = fusion_method
        
        # Metadata kodlayıcı ağ
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Füzyon yöntemine göre sınıflandırıcıyı tanımla
        if fusion_method == 'concat':
            # Basit birleştirme füzyonu
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim + 128, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'attention':
            # Dikkat mekanizması ile füzyon
            self.attention_module = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, self.feature_dim),
                nn.Sigmoid()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'gating':
            # Geçit mekanizması ile füzyon
            self.gate_module = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, self.feature_dim),
                nn.Sigmoid()
            )
            
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'film':
            # FiLM (Feature-wise Linear Modulation) füzyonu
            # Gamma ve beta parametrelerini oluştur (affine dönüşüm)
            self.gamma_layer = nn.Linear(128, self.feature_dim)
            self.beta_layer = nn.Linear(128, self.feature_dim)
            
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        else:
            raise ValueError(f"Desteklenmeyen füzyon yöntemi: {fusion_method}")
    
    def forward(self, images, metadata):
        """
        Forward pass
        
        Args:
            images: Görüntü tensor'ları [batch_size, channels, height, width]
            metadata: Metadata tensor'ları [batch_size, metadata_dim]
        
        Returns:
            logits: Sınıf logit'leri [batch_size, num_classes]
        """
        # Görüntü özniteliklerini çıkar
        image_features = self.backbone(images)
        
        # Metadata'yı kodla
        metadata_features = self.metadata_encoder(metadata)
        
        # Füzyon yöntemine göre birleştirme işlemi
        if self.fusion_method == 'concat':
            # Basit birleştirme füzyonu
            combined_features = torch.cat((image_features, metadata_features), dim=1)
            logits = self.classifier(combined_features)
            
        elif self.fusion_method == 'attention':
            # Dikkat mekanizması ile füzyon
            attention_weights = self.attention_module(metadata_features)
            attended_features = image_features * attention_weights
            logits = self.classifier(attended_features)
            
        elif self.fusion_method == 'gating':
            # Geçit mekanizması ile füzyon
            gates = self.gate_module(metadata_features)
            gated_features = image_features * gates
            logits = self.classifier(gated_features)
            
        elif self.fusion_method == 'film':
            # FiLM füzyonu
            gamma = self.gamma_layer(metadata_features)
            beta = self.beta_layer(metadata_features)
            
            # Affine dönüşüm: y = gamma * x + beta
            modulated_features = gamma * image_features + beta
            logits = self.classifier(modulated_features)
        
        return logits

# Farklı füzyon yöntemlerini açıklayan yardımcı bilgi
FUSION_METHODS = {
    'concat': 'Görüntü ve metadata özniteliklerini doğrudan birleştirir',
    'attention': 'Metadata bilgilerine dayalı olarak görüntü özniteliklerini ağırlıklandırır',
    'gating': 'Metadata bilgilerine dayalı olarak görüntü özniteliklerini filtreler',
    'film': 'Feature-wise Linear Modulation ile metadata bilgilerini görüntü özniteliklerine katar'
}

def get_fusion_method_description(method_name):
    """Füzyon yöntemi açıklamasını döndürür"""
    return FUSION_METHODS.get(method_name, 'Bilinmeyen füzyon yöntemi')