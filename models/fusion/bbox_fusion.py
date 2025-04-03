# models/fusion/bbox_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class BBoxFusionModel(nn.Module):
    """
    Bounding box entegrasyonu için model sınıfı.
    İki farklı modu destekler:
    1. Global-ROI füzyonu: Global görüntü öznitelikleri ve ROI özniteliklerini birleştirir
    2. Auxiliary loss: Hastalık sınıflandırma ve bounding box tahmini için çok görevli öğrenme
    """
    
    def __init__(self, backbone, num_classes, bbox_mode='fusion', use_roi_pool=True):
        """
        Args:
            backbone: Görüntü öznitelik çıkarıcı
            num_classes: Sınıf sayısı
            bbox_mode: 'fusion' veya 'auxiliary'
            use_roi_pool: ROI Pooling kullanılsın mı? (Sadece fusion modunda)
        """
        super(BBoxFusionModel, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.bbox_mode = bbox_mode
        self.use_roi_pool = use_roi_pool
        
        # Backbone tipini kontrol et
        self.backbone_type = type(backbone).__name__
        
        # ROI Pooling veya ROI Align katmanı (fusion modunda)
        if bbox_mode == 'fusion' and use_roi_pool:
            self.roi_layer = ops.RoIPool(output_size=(7, 7), spatial_scale=1/32.0)
            # Not: spatial_scale değeri, girdi görüntü boyutunun öznitelik haritası 
            # boyutuna oranını belirtir. ResNet50 için bu oran genellikle 1/32'dir.
            
            # ROI özellikleri için tam bağlantılı katman
            self.roi_fc = nn.Sequential(
                nn.Linear(self.feature_dim * 7 * 7, 1024),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Global ve ROI özelliklerini birleştiren sınıflandırıcı
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim + 1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        else:
            # Standart sınıflandırıcı (ROI pooling olmadan)
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        # Auxiliary bbox regresyonu (auxiliary modunda)
        if bbox_mode == 'auxiliary':
            self.bbox_regressor = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 4)  # (x, y, w, h) tahmini
            )
            
            # Bbox var/yok sınıflandırıcısı
            self.bbox_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)  # Bounding box var mı? (ikili sınıflandırma)
            )
    
    def _prepare_boxes(self, boxes, image_size=224):
        """Bounding box tensörlerini ROI pooling için formatlar"""
        batch_size = boxes.size(0)
        
        # ROI pooling için gerekli format: (batch_idx, x1, y1, x2, y2)
        formatted_boxes = []
        
        for i in range(batch_size):
            # Normalized (x, y, w, h) formatını (x1, y1, x2, y2) formatına dönüştür
            x, y, w, h = boxes[i] * image_size  # Normalize değerleri piksel koordinatlarına dönüştür
            
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            box = torch.tensor([i, x1, y1, x2, y2], dtype=torch.float, device=boxes.device)
            formatted_boxes.append(box)
        
        return torch.stack(formatted_boxes)
    
    def _get_feature_maps(self, images):
        """
        Farklı omurga tiplerinden öznitelik haritalarını alır
        """
        if hasattr(self.backbone, 'features'):
            # Doğrudan features özniteliği varsa kullan
            return self.backbone.features(images)
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'features'):
            # Sarmalanmış modelde features özniteliği varsa kullan
            return self.backbone.model.features(images)
        elif self.backbone_type == 'ResNetBackbone':
            # ResNet için özel durum - son avgpool öncesindeki öznitelik haritaları
            x = self.backbone.model.conv1(images)
            x = self.backbone.model.bn1(x)
            x = self.backbone.model.relu(x)
            x = self.backbone.model.maxpool(x)
            
            x = self.backbone.model.layer1(x)
            x = self.backbone.model.layer2(x)
            x = self.backbone.model.layer3(x)
            x = self.backbone.model.layer4(x)
            
            return x
        else:
            # Desteklenmeyen omurga tipi
            raise ValueError(f"Desteklenmeyen backbone tipi: {self.backbone_type}. 'features' özniteliği bulunamadı.")
    
    def forward(self, images, boxes=None, has_bbox=None):
        """
        Forward pass
        
        Args:
            images: Görüntü tensor'ları [batch_size, channels, height, width]
            boxes: Bounding box tensor'ları [batch_size, 4] formatında (x, y, w, h) - normalize edilmiş
            has_bbox: Bounding box var mı? [batch_size] formatında boolean tensor
        
        Returns:
            dict: Çıktı sözlüğü (logits, bbox_pred, bbox_present_prob)
        """
        # Görüntü özniteliklerini çıkar
        if self.bbox_mode == 'fusion' and self.use_roi_pool:
            # Öznitelik haritalarını al (backbone tipine bağlı olarak)
            feature_maps = self._get_feature_maps(images)
            
            # Global öznitelikleri al
            global_features = self.backbone(images)  # Global havuzlanmış öznitelikler
        else:
            # Sadece global öznitelikleri al
            global_features = self.backbone(images)
        
        outputs = {}
        
        if self.bbox_mode == 'fusion' and self.use_roi_pool and boxes is not None and has_bbox is not None:
            # ROI pooling ile öznitelik çıkarma
            valid_boxes_mask = has_bbox > 0.5  # Geçerli bbox'ları belirle
            
            if valid_boxes_mask.sum() > 0:
                # Geçerli bbox'ları seç
                valid_boxes = boxes[valid_boxes_mask]
                
                # BoxList formatına dönüştür
                roi_boxes = self._prepare_boxes(valid_boxes)
                
                # ROI pooling uygula
                roi_features = self.roi_layer(feature_maps, roi_boxes)
                roi_features = roi_features.view(roi_features.size(0), -1)  # Düzleştir
                
                # ROI özellikleri için FC katmanını uygula
                roi_features = self.roi_fc(roi_features)
                
                # Global öznitelikler ve ROI özniteliklerini birleştir (geçerli bbox'lar için)
                valid_global_features = global_features[valid_boxes_mask]
                valid_combined_features = torch.cat((valid_global_features, roi_features), dim=1)
                
                # Geçerli bbox'lar için sınıflandırma
                valid_logits = self.classifier(valid_combined_features)
                
                # Tüm batch için sonuçları doldur
                logits = torch.zeros(images.size(0), self.classifier[-1].out_features, 
                                    dtype=torch.float, device=images.device)
                logits[valid_boxes_mask] = valid_logits
                
                # Bbox olmayan örnekler için sadece global öznitelikleri kullan
                if (~valid_boxes_mask).sum() > 0:
                    invalid_global_features = global_features[~valid_boxes_mask]
                    # Boyut uyumsuzluğu olduğu için geçici bir sınıflandırıcı oluştur
                    temp_classifier = nn.Linear(self.feature_dim, self.classifier[-1].out_features).to(images.device)
                    invalid_logits = temp_classifier(invalid_global_features)
                    logits[~valid_boxes_mask] = invalid_logits
            else:
                # Hiç geçerli bbox yoksa sadece global öznitelikleri kullan
                # Boyut uyumsuzluğu olduğu için geçici bir sınıflandırıcı oluştur
                temp_classifier = nn.Linear(self.feature_dim, self.classifier[-1].out_features).to(images.device)
                logits = temp_classifier(global_features)
                
            outputs['logits'] = logits
            
        else:
            # Standart sınıflandırma (ROI pooling olmadan)
            logits = self.classifier(global_features)
            outputs['logits'] = logits
        
        # Auxiliary bounding box regresyonu
        if self.bbox_mode == 'auxiliary':
            bbox_pred = self.bbox_regressor(global_features)
            bbox_present_prob = self.bbox_classifier(global_features)
            
            outputs['bbox_pred'] = bbox_pred
            outputs['bbox_present_prob'] = bbox_present_prob
        
        return outputs

# Auxiliary loss hesaplama
def bbox_auxiliary_loss(bbox_pred, bbox_true, bbox_present_prob, has_bbox, lambda_reg=1.0, lambda_cls=0.5):
    """
    Bounding box auxiliary loss hesaplama
    
    Args:
        bbox_pred: Tahmin edilen bbox [batch_size, 4]
        bbox_true: Gerçek bbox [batch_size, 4]
        bbox_present_prob: Bbox varlığı olasılığı [batch_size, 1]
        has_bbox: Gerçek bbox varlığı bayrağı [batch_size]
        lambda_reg: Regresyon kaybı ağırlığı
        lambda_cls: Sınıflandırma kaybı ağırlığı
    
    Returns:
        toplam_kayıp: Ağırlıklı toplam kayıp
    """
    # Sınıflandırma kaybı (bbox var mı?)
    cls_loss = F.binary_cross_entropy_with_logits(
        bbox_present_prob.squeeze(), has_bbox.float())
    
    # Regresyon kaybı (sadece geçerli bbox'lar için)
    if has_bbox.sum() > 0:
        # Smooth L1 kaybı
        valid_mask = has_bbox > 0.5
        valid_bbox_pred = bbox_pred[valid_mask]
        valid_bbox_true = bbox_true[valid_mask]
        
        reg_loss = F.smooth_l1_loss(valid_bbox_pred, valid_bbox_true)
    else:
        reg_loss = 0.0
    
    # Toplam kayıp
    total_loss = lambda_cls * cls_loss
    if has_bbox.sum() > 0:
        total_loss += lambda_reg * reg_loss
    
    return total_loss