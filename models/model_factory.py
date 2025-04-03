# models/model_factory.py

import torch
import torch.nn as nn
from models.backbones.resnet import ResNetBackbone
from models.backbones.vgg import VGGBackbone
from models.backbones.inception import InceptionBackbone
from models.backbones.mobilenet import MobileNetBackbone
from models.backbones.alexnet import AlexNetBackbone
from models.fusion.metadata_fusion import MetadataFusionModel
from models.fusion.bbox_fusion import BBoxFusionModel
from models.fusion.hybrid_models import HybridChestXpertModel

class ModelFactory:
    """
    ChestXpert projesi için model oluşturma fabrikası.
    Farklı omurga mimarilerini ve füzyon modellerini kolayca oluşturmak için kullanılır.
    """
    
    @staticmethod
    def get_backbone(name, **kwargs):
        """
        İstenen omurga modelini oluşturur.
        
        Args:
            name: Omurga modeli adı
                - 'resnet18', 'resnet50', 'resnet101'
                - 'vgg16', 'vgg19'
                - 'inception_v3'
                - 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
                - 'alexnet'
            **kwargs: Model oluşturma parametreleri
                - pretrained: Önceden eğitilmiş ağırlıkları kullan (varsayılan: True)
                - freeze_layers: İlk katmanları dondur (varsayılan: True)
                
        Returns:
            model: Oluşturulan omurga modeli
        """
        # Varsayılan parametreleri belirle
        pretrained = kwargs.get('pretrained', True)
        freeze_layers = kwargs.get('freeze_layers', True)
        
        # ResNet modelleri
        if name == 'resnet18' or name == 'resnet50' or name == 'resnet101':
            return ResNetBackbone(variant=name, pretrained=pretrained, freeze_layers=freeze_layers)
        
        # VGG modelleri
        elif name == 'vgg16' or name == 'vgg19':
            return VGGBackbone(variant=name, pretrained=pretrained, freeze_layers=freeze_layers)
        
        # Inception modeli
        elif name == 'inception_v3':
            aux_logits = kwargs.get('aux_logits', False)
            return InceptionBackbone(pretrained=pretrained, freeze_layers=freeze_layers, aux_logits=aux_logits)
        
        # MobileNet modelleri
        elif name in ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large']:
            return MobileNetBackbone(variant=name, pretrained=pretrained, freeze_layers=freeze_layers)
        
        # AlexNet modeli
        elif name == 'alexnet':
            return AlexNetBackbone(pretrained=pretrained, freeze_layers=freeze_layers)
        
        else:
            raise ValueError(f"Desteklenmeyen omurga modeli: {name}")
    
    @staticmethod
    def get_metadata_model(backbone_name, num_classes, metadata_dim=3, fusion_method='concat', **kwargs):
        """
        Metadata füzyon modeli oluşturur.
        
        Args:
            backbone_name: Omurga modeli adı
            num_classes: Sınıf sayısı
            metadata_dim: Metadata öznitelik boyutu
            fusion_method: Füzyon yöntemi ('concat', 'attention', 'gating', 'film')
            **kwargs: Diğer parametreler
                
        Returns:
            model: Metadata füzyon modeli
        """
        # Omurga modelini oluştur
        backbone = ModelFactory.get_backbone(backbone_name, **kwargs)
        
        # Metadata modeli oluştur
        return MetadataFusionModel(backbone, num_classes, metadata_dim, fusion_method)
    
    @staticmethod
    def get_bbox_model(backbone_name, num_classes, bbox_mode='fusion', use_roi_pool=True, **kwargs):
        """
        Bounding box füzyon modeli oluşturur.
        
        Args:
            backbone_name: Omurga modeli adı
            num_classes: Sınıf sayısı
            bbox_mode: 'fusion' veya 'auxiliary'
            use_roi_pool: ROI Pooling kullanılsın mı?
            **kwargs: Diğer parametreler
                
        Returns:
            model: Bounding box füzyon modeli
        """
        # Omurga modelini oluştur
        backbone = ModelFactory.get_backbone(backbone_name, **kwargs)
        
        # Bounding box modeli oluştur
        return BBoxFusionModel(backbone, num_classes, bbox_mode, use_roi_pool)
    
    @staticmethod
    def get_hybrid_model(backbone_name, num_classes, metadata_dim=3, fusion_method='attention', 
                        use_bbox=True, use_roi=False, **kwargs):
        """
        Hibrit model oluşturur (Görüntü + Metadata + BBox).
        
        Args:
            backbone_name: Omurga modeli adı
            num_classes: Sınıf sayısı
            metadata_dim: Metadata öznitelik boyutu
            fusion_method: Füzyon yöntemi ('concat', 'attention', 'cross_attention', 'film')
            use_bbox: Bounding box bilgisini kullan
            use_roi: ROI pooling/ROI align kullan
            **kwargs: Diğer parametreler
                
        Returns:
            model: Hibrit model
        """
        # Omurga modelini oluştur
        backbone = ModelFactory.get_backbone(backbone_name, **kwargs)
        
        # Hibrit modeli oluştur
        return HybridChestXpertModel(
            backbone, num_classes, metadata_dim, fusion_method, use_bbox, use_roi
        )
    
    @staticmethod
    def create_model_from_config(config):
        """
        Konfigürasyon sözlüğüne göre model oluşturur.
        
        Args:
            config: Model konfigürasyon sözlüğü
                {
                    'model_type': 'backbone | metadata | bbox | hybrid',
                    'backbone': 'resnet50 | vgg16 | ...',
                    'num_classes': 15,
                    'pretrained': True,
                    'freeze_backbone': True,
                    'metadata': {
                        'use_metadata': True,
                        'metadata_dim': 3,
                        'fusion_method': 'concat'
                    },
                    'bbox': {
                        'use_bbox': True,
                        'use_roi': True,
                        'bbox_mode': 'fusion'
                    },
                    'fusion': {
                        'method': 'attention'
                    }
                }
                
        Returns:
            model: Oluşturulan model
        """
        model_type = config.get('model_type', 'backbone')
        backbone = config.get('backbone', 'resnet50')
        num_classes = config.get('num_classes', 15)
        pretrained = config.get('pretrained', True)
        freeze_backbone = config.get('freeze_backbone', True)
        
        if model_type == 'backbone':
            # Sadece omurga modeli
            return ModelFactory.get_backbone(
                backbone, 
                pretrained=pretrained, 
                freeze_layers=freeze_backbone
            )
            
        elif model_type == 'metadata':
            # Metadata füzyon modeli
            metadata_config = config.get('metadata', {})
            metadata_dim = metadata_config.get('metadata_dim', 3)
            fusion_method = metadata_config.get('fusion_method', 'concat')
            
            return ModelFactory.get_metadata_model(
                backbone, 
                num_classes, 
                metadata_dim, 
                fusion_method,
                pretrained=pretrained, 
                freeze_layers=freeze_backbone
            )
            
        elif model_type == 'bbox':
            # Bounding box füzyon modeli
            bbox_config = config.get('bbox', {})
            bbox_mode = bbox_config.get('bbox_mode', 'fusion')
            use_roi_pool = bbox_config.get('use_roi', True)
            
            return ModelFactory.get_bbox_model(
                backbone, 
                num_classes, 
                bbox_mode, 
                use_roi_pool,
                pretrained=pretrained, 
                freeze_layers=freeze_backbone
            )
            
        elif model_type == 'hybrid':
            # Hibrit model
            metadata_config = config.get('metadata', {})
            bbox_config = config.get('bbox', {})
            fusion_config = config.get('fusion', {})
            
            metadata_dim = metadata_config.get('metadata_dim', 3)
            use_bbox = bbox_config.get('use_bbox', True)
            use_roi = bbox_config.get('use_roi', False)
            fusion_method = fusion_config.get('method', 'attention')
            
            return ModelFactory.get_hybrid_model(
                backbone, 
                num_classes, 
                metadata_dim, 
                fusion_method, 
                use_bbox, 
                use_roi,
                pretrained=pretrained, 
                freeze_layers=freeze_backbone
            )
            
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}")
    
    @staticmethod
    def list_available_backbones():
        """Kullanılabilir tüm omurga modellerini listeler"""
        return {
            'resnet': ['resnet18', 'resnet50', 'resnet101'],
            'vgg': ['vgg16', 'vgg19'],
            'inception': ['inception_v3'],
            'mobilenet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
            'alexnet': ['alexnet']
        }
    
    @staticmethod
    def list_available_fusion_methods():
        """Kullanılabilir tüm füzyon yöntemlerini listeler"""
        return {
            'metadata': ['concat', 'attention', 'gating', 'film'],
            'bbox': ['fusion', 'auxiliary'],
            'hybrid': ['concat', 'attention', 'cross_attention', 'film']
        }

# Test için
if __name__ == "__main__":
    # Kullanılabilir omurga modellerini listele
    print("Kullanılabilir omurga modelleri:")
    backbones = ModelFactory.list_available_backbones()
    for family, models in backbones.items():
        print(f"  {family}: {', '.join(models)}")
    
    # Kullanılabilir füzyon yöntemlerini listele
    print("\nKullanılabilir füzyon yöntemleri:")
    fusion_methods = ModelFactory.list_available_fusion_methods()
    for model_type, methods in fusion_methods.items():
        print(f"  {model_type}: {', '.join(methods)}")
    
    # Konfigürasyon örneği
    config = {
        'model_type': 'hybrid',
        'backbone': 'resnet50',
        'num_classes': 15,
        'pretrained': True,
        'freeze_backbone': True,
        'metadata': {
            'use_metadata': True,
            'metadata_dim': 3
        },
        'bbox': {
            'use_bbox': True,
            'use_roi': True
        },
        'fusion': {
            'method': 'attention'
        }
    }
    
    # Konfigürasyona göre model oluştur
    model = ModelFactory.create_model_from_config(config)
    print(f"\nOluşturulan model türü: {type(model).__name__}")
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    metadata = torch.randn(4, 3)
    boxes = torch.rand(4, 4)  # normalized bbox coordinates
    has_bbox = torch.randint(0, 2, (4,)).float()  # binary flag
    
    # Forward pass
    outputs = model(x, metadata, boxes, has_bbox)
    
    # Çıktıları kontrol et
    print("\nModel çıktıları:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")