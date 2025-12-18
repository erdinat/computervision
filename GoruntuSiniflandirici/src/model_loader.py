"""
Model mimarisi ve yükleme fonksiyonları
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List
import os


class ImageClassifier(nn.Module):
    """
    Transfer learning kullanarak görüntü sınıflandırma modeli.
    ResNet18 mimarisini kullanır ve önceden eğitilmiş ağırlıkları yükler.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        ImageClassifier modelini başlatır.
        
        Args:
            num_classes: Sınıf sayısı
            pretrained: Önceden eğitilmiş ağırlıkları kullan (ImageNet)
        """
        super(ImageClassifier, self).__init__()
        
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Son fully connected katmanını değiştir
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        return self.backbone(x)


def create_model(num_classes: int, pretrained: bool = True) -> ImageClassifier:
    """
    Yeni bir model oluşturur.
    
    Args:
        num_classes: Sınıf sayısı
        pretrained: Önceden eğitilmiş ağırlıkları kullan
        
    Returns:
        ImageClassifier modeli
    """
    model = ImageClassifier(num_classes=num_classes, pretrained=pretrained)
    return model


def load_model(
    model_path: str,
    num_classes: int,
    device: str = 'cpu'
) -> ImageClassifier:
    """
    Eğitilmiş modeli diskten yükler.
    
    Args:
        model_path: Model dosyasının yolu (.pth)
        num_classes: Sınıf sayısı
        device: Cihaz ('cpu' veya 'cuda')
        
    Returns:
        Yüklenmiş ve eval moduna alınmış model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def save_model(model: ImageClassifier, model_path: str):
    """
    Modeli diske kaydeder.
    
    Args:
        model: Kaydedilecek model
        model_path: Kayıt yolu
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model kaydedildi: {model_path}")


def predict(
    model: ImageClassifier,
    image_tensor: torch.Tensor,
    class_names: List[str],
    device: str = 'cpu',
    top_k: int = 5
) -> List[dict]:
    """
    Görüntü için tahmin yapar.
    
    Args:
        model: Eğitilmiş model
        image_tensor: Ön işleme tabi tutulmuş görüntü tensor'ı (C, H, W) veya (B, C, H, W)
        class_names: Sınıf isimlerinin listesi
        device: Cihaz
        top_k: En yüksek k tahmin göster
        
    Returns:
        Tahmin sonuçlarının listesi [{'class': str, 'probability': float}, ...]
    """
    model.eval()
    
    # Batch dimension ekle eğer yoksa
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    results = []
    for i in range(top_k):
        class_idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        results.append({
            'class': class_names[class_idx],
            'probability': prob
        })
    
    return results

