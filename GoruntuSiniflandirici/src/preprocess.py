"""
Ön işleme modülü - Görüntülerin normalize edilmesi, yeniden boyutlandırılması ve augmentation
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Görüntü ön işleme sınıfı.
    Görüntüleri normalize eder, yeniden boyutlandırır ve augmentation uygular.
    
    Attributes:
        image_size (Tuple[int, int]): Hedef görüntü boyutu (width, height)
        mean (Tuple[float, float, float]): Normalizasyon için ortalama değerleri
        std (Tuple[float, float, float]): Normalizasyon için standart sapma değerleri
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        ImagePreprocessor sınıfını başlatır.
        
        Args:
            image_size: Hedef görüntü boyutu (width, height)
            mean: RGB kanalları için ortalama değerleri (ImageNet standartları)
            std: RGB kanalları için standart sapma değerleri (ImageNet standartları)
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        
        # Test/Inference için transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size[1], image_size[0])),
            transforms.CenterCrop(image_size[1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Eğitim için augmentation içeren transform pipeline
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size[1] + 32, image_size[0] + 32)),
            transforms.RandomCrop(image_size[1]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def preprocess_image(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """
        Tek bir görüntüyü ön işleme tabi tutar.
        
        Time Complexity: O(H*W) - H ve W görüntü boyutları
        
        Args:
            image: PIL Image nesnesi
            augment: Eğer True ise augmentation uygulanır (eğitim için)
            
        Returns:
            Ön işleme tabi tutulmuş tensor (C, H, W) formatında
        """
        if augment:
            return self.train_transform(image)
        return self.transform(image)
    
    def preprocess_batch(
        self,
        images: list,
        augment: bool = False
    ) -> torch.Tensor:
        """
        Birden fazla görüntüyü batch olarak ön işleme tabi tutar.
        
        Time Complexity: O(B*H*W) - B batch size, H ve W görüntü boyutları
        
        Args:
            images: PIL Image nesnelerinin listesi
            augment: Eğer True ise augmentation uygulanır
            
        Returns:
            Batch tensor (B, C, H, W) formatında
        """
        processed_images = [self.preprocess_image(img, augment) for img in images]
        return torch.stack(processed_images)
    
    def load_and_preprocess(self, image_path: str, augment: bool = False) -> torch.Tensor:
        """
        Görüntüyü dosyadan yükler ve ön işleme tabi tutar.
        
        Args:
            image_path: Görüntü dosyasının yolu
            augment: Eğer True ise augmentation uygulanır
            
        Returns:
            Ön işleme tabi tutulmuş tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.preprocess_image(image, augment)
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Normalize edilmiş tensor'ı orijinal görüntü formatına geri döndürür.
        Görselleştirme için kullanılır.
        
        Args:
            tensor: Normalize edilmiş tensor (C, H, W) veya (B, C, H, W)
            
        Returns:
            Denormalize edilmiş numpy array (H, W, C) veya (B, H, W, C)
        """
        if tensor.dim() == 4:  # Batch
            tensor = tensor.clone()
            for i in range(tensor.size(0)):
                for t, m, s in zip(tensor[i], self.mean, self.std):
                    t.mul_(s).add_(m)
            tensor = torch.clamp(tensor, 0, 1)
            return tensor.permute(0, 2, 3, 1).cpu().numpy()
        else:  # Single image
            tensor = tensor.clone()
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            tensor = torch.clamp(tensor, 0, 1)
            return tensor.permute(1, 2, 0).cpu().numpy()
