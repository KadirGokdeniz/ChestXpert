# models/fusion/hybrid_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class HybridChestXpertModel(nn.Module):
    """
    Görüntü, metadata ve bounding box verilerini birleştiren hibrit model.
    
    Bu model:
    1. Görüntü özniteliklerini çıkarmak için bir CNN omurgası kullanır
    2. Metadata bilgilerini (yaş, cinsiyet, görüntüleme pozisyonu) işler
    3. Mevcut olduğunda bounding box bilgilerini kullanır
    4. Bunları çeşitli füzyon stratejileri ile birleştirir
    """
    
    def __init__(self, backbone, num_classes=15, metadata_dim=3, 
                 fusion_method='attention', use_bbox=True, use_roi=False):
        """
        Args:
            backbone: Görüntü öznitelik çıkarıcı omurga
            num_classes: Sınıf sayısı
            metadata_dim: Metadata öznitelik boyutu (yaş, cinsiyet, vb.)
            fusion_method: Füzyon yöntemi ('concat', 'attention', 'cross_attention', 'film')
            use_bbox: Bounding box bilgisini kullan
            use_roi: ROI pooling/ROI align kullan (True ise tam omurga özellikleri gerekir)
        """
        super(HybridChestXpertModel, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim
        self.fusion_method = fusion_method
        self.use_bbox = use_bbox
        self.use_roi = use_roi
        self.num_classes = num_classes
        
        # Backbone tipini kontrol et
        self.backbone_type = type(backbone).__name__
        
        # Metadata kodlayıcı
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Bounding box kodlayıcı
        if use_bbox:
            # bbox var/yok bilgisiyle birlikte 5 boyutlu girdi (x, y, w, h, has_bbox)
            self.bbox_encoder = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128)
            )
            
            # ROI Pooling kullanılacaksa
            if use_roi:
                self.roi_layer = ops.RoIPool(output_size=(7, 7), spatial_scale=1/32.0)
                # ROI özellikleri için tam bağlantılı katman
                self.roi_fc = nn.Sequential(
                    nn.Linear(self.feature_dim * 7 * 7, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.ReLU()
                )
        
        # Füzyon yöntemine göre katmanları oluştur
        if fusion_method == 'concat':
            # Girdi boyutunu hesapla
            input_dim = self.feature_dim + 128  # CNN + metadata
            if use_bbox:
                input_dim += 128  # + bbox
                if use_roi:
                    input_dim += 512  # + ROI features
            
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'attention':
            # Metadata attention
            self.metadata_attention = nn.Sequential(
                nn.Linear(128, 256),
                nn.Tanh(),
                nn.Linear(256, self.feature_dim),
                nn.Sigmoid()
            )
            
            # Bbox attention (eğer kullanılıyorsa)
            if use_bbox:
                self.bbox_attention = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.Tanh(),
                    nn.Linear(256, self.feature_dim),
                    nn.Sigmoid()
                )
            
            # Classifier
            classifier_input_dim = self.feature_dim
            if use_roi:
                classifier_input_dim += 512  # ROI features
                
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'cross_attention':
            # Çapraz dikkat için öznitelik boyutu
            self.query_dim = 256
            
            # Query, key, value projeksiyonları
            self.query_proj = nn.Linear(self.feature_dim, self.query_dim)
            self.metadata_key_proj = nn.Linear(128, self.query_dim)
            self.metadata_value_proj = nn.Linear(128, self.query_dim)
            
            if use_bbox:
                self.bbox_key_proj = nn.Linear(128, self.query_dim)
                self.bbox_value_proj = nn.Linear(128, self.query_dim)
            
            # Çıktı projeksiyonu
            self.output_proj = nn.Linear(self.query_dim, self.feature_dim)
            
            # Classifier
            classifier_input_dim = self.feature_dim
            if use_roi:
                classifier_input_dim += 512  # ROI features
                
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif fusion_method == 'film':
            # FiLM (Feature-wise Linear Modulation)
            self.metadata_gamma = nn.Linear(128, self.feature_dim)
            self.metadata_beta = nn.Linear(128, self.feature_dim)
            
            if use_bbox:
                self.bbox_gamma = nn.Linear(128, self.feature_dim)
                self.bbox_beta = nn.Linear(128, self.feature_dim)
            
            # Classifier
            classifier_input_dim = self.feature_dim
            if use_roi:
                classifier_input_dim += 512  # ROI features
                
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        else:
            raise ValueError(f"Desteklenmeyen füzyon yöntemi: {fusion_method}")
        
        # Auxiliary task: Bounding box regresyonu
        if use_bbox:
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
                nn.Linear(128, 1)  # Bounding box var mı?
            )
    
    def _prepare_boxes(self, boxes, has_bbox, image_size=224):
        """Bounding box tensörlerini ROI pooling için formatlar"""
        batch_size = boxes.size(0)
        
        # Geçerli bbox'ları belirle
        valid_indices = torch.nonzero(has_bbox > 0.5).squeeze()
        
        if valid_indices.numel() == 0:
            return None
        
        # ROI pooling için gerekli format: (batch_idx, x1, y1, x2, y2)
        formatted_boxes = []
        
        for i in valid_indices:
            # Normalize (x, y, w, h) formatını (x1, y1, x2, y2) formatına dönüştür
            x, y, w, h = boxes[i] * image_size  # Normalize değerleri piksel koordinatlarına dönüştür
            
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            # Batch indeksi ve koordinatları birleştir
            box = torch.tensor([i, x1, y1, x2, y2], dtype=torch.float, device=boxes.device)
            formatted_boxes.append(box)
        
        if len(formatted_boxes) == 0:
            return None
            
        return torch.stack(formatted_boxes)
    
    def _cross_attention(self, queries, keys, values):
        """Çapraz dikkat mekanizması uygular"""
        # Queries: [batch_size, query_dim]
        # Keys: [batch_size, key_dim]
        # Values: [batch_size, value_dim]
        
        # Attention scores
        scores = torch.matmul(queries.unsqueeze(1), keys.unsqueeze(2))  # [batch_size, 1, 1]
        scores = scores / torch.sqrt(torch.tensor(self.query_dim, dtype=torch.float, device=scores.device))
        
        # Attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, 1, 1]
        
        # Weighted sum
        weighted_values = weights * values.unsqueeze(1)  # [batch_size, 1, value_dim]
        
        return weighted_values.squeeze(1)  # [batch_size, value_dim]
    
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
    
    def forward(self, images, metadata=None, boxes=None, has_bbox=None):
        """
        Forward pass
        
        Args:
            images: Görüntü tensörleri [batch_size, channels, height, width]
            metadata: Metadata tensörleri [batch_size, metadata_dim]
            boxes: Bounding box tensörleri [batch_size, 4] (x, y, w, h) - normalize edilmiş
            has_bbox: Bounding box var mı? [batch_size] formatında boolean tensor
        
        Returns:
            dict: Çıktı sözlüğü {
                'logits': Sınıf logitleri,
                'bbox_pred': Tahmin edilen bounding box (opsiyonel),
                'bbox_present_prob': Bbox varlık olasılığı (opsiyonel),
                'attention_weights': Dikkat ağırlıkları (opsiyonel)
            }
        """
        outputs = {}
        
        # Görüntü özniteliklerini çıkar
        if self.use_roi:
            # Öznitelik haritalarını ve havuzlanmış öznitelikleri ayrı ayrı al
            feature_maps = self._get_feature_maps(images)  # Öznitelik haritaları
            img_features = self.backbone(images)  # Global havuzlanmış öznitelikler
        else:
            # Sadece global havuzlanmış öznitelikleri al
            img_features = self.backbone(images)
        
        # Metadata özniteliklerini çıkar (varsa)
        if metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
        else:
            metadata_features = None
        
        # Bbox özniteliklerini çıkar (varsa)
        if self.use_bbox and boxes is not None and has_bbox is not None:
            # Bbox bilgisini ve has_bbox bayrağını birleştir
            bbox_info = torch.cat((boxes, has_bbox.unsqueeze(1)), dim=1)
            bbox_features = self.bbox_encoder(bbox_info)
            
            # ROI özniteliklerini çıkar (varsa)
            if self.use_roi:
                roi_boxes = self._prepare_boxes(boxes, has_bbox)
                
                if roi_boxes is not None:
                    # ROI pooling uygula
                    roi_features_map = self.roi_layer(feature_maps, roi_boxes)
                    roi_features_flat = roi_features_map.view(roi_features_map.size(0), -1)  # Düzleştir
                    roi_features = self.roi_fc(roi_features_flat)
                    
                    # ROI özniteliklerini batch indekslerine göre doldur
                    batch_roi_features = torch.zeros(
                        images.size(0), 512, dtype=torch.float, device=images.device)
                    
                    for i, box_entry in enumerate(roi_boxes):
                        batch_idx = int(box_entry[0])
                        batch_roi_features[batch_idx] = roi_features[i]
                else:
                    batch_roi_features = torch.zeros(
                        images.size(0), 512, dtype=torch.float, device=images.device)
            else:
                batch_roi_features = None
        else:
            bbox_features = None
            batch_roi_features = None
        
        # Füzyon yöntemine göre öznitelikleri birleştir
        if self.fusion_method == 'concat':
            # Features to concatenate
            features_list = [img_features]
            
            if metadata_features is not None:
                features_list.append(metadata_features)
                
            if bbox_features is not None:
                features_list.append(bbox_features)
                
            if batch_roi_features is not None:
                features_list.append(batch_roi_features)
            
            # Concat all available features
            combined_features = torch.cat(features_list, dim=1)
            
            # Final classification
            logits = self.classifier(combined_features)
            
        elif self.fusion_method == 'attention':
            # Start with image features
            attended_features = img_features
            
            # Apply metadata attention if available
            if metadata_features is not None:
                metadata_weights = self.metadata_attention(metadata_features)
                attended_features = attended_features * metadata_weights
                
                # Store attention weights for visualization
                outputs['metadata_attention'] = metadata_weights
            
            # Apply bbox attention if available
            if bbox_features is not None:
                bbox_weights = self.bbox_attention(bbox_features)
                attended_features = attended_features * bbox_weights
                
                # Store attention weights for visualization
                outputs['bbox_attention'] = bbox_weights
            
            # Combine with ROI features if available
            if batch_roi_features is not None:
                features_list = [attended_features, batch_roi_features]
                final_features = torch.cat(features_list, dim=1)
            else:
                final_features = attended_features
            
            # Final classification
            logits = self.classifier(final_features)
            
        elif self.fusion_method == 'cross_attention':
            # Project image features to query space
            queries = self.query_proj(img_features)
            
            # Process metadata through cross-attention if available
            if metadata_features is not None:
                meta_keys = self.metadata_key_proj(metadata_features)
                meta_values = self.metadata_value_proj(metadata_features)
                
                # Apply cross-attention
                meta_context = self._cross_attention(queries, meta_keys, meta_values)
                
                # Update queries with metadata context
                queries = queries + meta_context
            
            # Process bbox through cross-attention if available
            if bbox_features is not None:
                bbox_keys = self.bbox_key_proj(bbox_features)
                bbox_values = self.bbox_value_proj(bbox_features)
                
                # Apply cross-attention
                bbox_context = self._cross_attention(queries, bbox_keys, bbox_values)
                
                # Update queries with bbox context
                queries = queries + bbox_context
            
            # Project back to feature space
            attended_features = self.output_proj(queries)
            
            # Combine with ROI features if available
            if batch_roi_features is not None:
                features_list = [attended_features, batch_roi_features]
                final_features = torch.cat(features_list, dim=1)
            else:
                final_features = attended_features
            
            # Final classification
            logits = self.classifier(final_features)
            
        elif self.fusion_method == 'film':
            # Start with image features
            modulated_features = img_features
            
            # Apply FiLM conditioning from metadata if available
            if metadata_features is not None:
                gamma_m = self.metadata_gamma(metadata_features)
                beta_m = self.metadata_beta(metadata_features)
                
                modulated_features = gamma_m * modulated_features + beta_m
            
            # Apply FiLM conditioning from bbox if available
            if bbox_features is not None:
                gamma_b = self.bbox_gamma(bbox_features)
                beta_b = self.bbox_beta(bbox_features)
                
                modulated_features = gamma_b * modulated_features + beta_b
            
            # Combine with ROI features if available
            if batch_roi_features is not None:
                features_list = [modulated_features, batch_roi_features]
                final_features = torch.cat(features_list, dim=1)
            else:
                final_features = modulated_features
            
            # Final classification
            logits = self.classifier(final_features)
        
        # Store the main output
        outputs['logits'] = logits
        
        # Auxiliary bbox outputs if enabled
        if self.use_bbox and boxes is not None:
            bbox_pred = self.bbox_regressor(img_features)
            bbox_present_prob = self.bbox_classifier(img_features)
            
            outputs['bbox_pred'] = bbox_pred
            outputs['bbox_present_prob'] = bbox_present_prob
        
        return outputs

# Örnek konfigürasyon
def example_hybrid_model_config():
    """Hibrit model için örnek konfigürasyon sözlüğünü döndürür"""
    return {
        'backbone': 'resnet50',
        'num_classes': 15,
        'freeze_backbone': True,
        'pretrained': True,
        'metadata': {
            'use_metadata': True,
            'metadata_dim': 3,  # age, gender, view_position
        },
        'bbox': {
            'use_bbox': True,
            'use_roi': True,
            'auxiliary_loss': True,
            'lambda_reg': 1.0,
            'lambda_cls': 0.5
        },
        'fusion': {
            'method': 'attention',  # 'concat', 'attention', 'cross_attention', 'film'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 50,
            'optimizer': 'adam',
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'warmup_epochs': 5
        }
    }