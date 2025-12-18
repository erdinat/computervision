"""
Model eğitim döngüsü ve yardımcı fonksiyonlar
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from typing import Dict, Optional
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from .preprocess import ImagePreprocessor
from .model_loader import create_model, save_model


class Trainer:
    """
    Model eğitimi için trainer sınıfı.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        num_epochs: int = 10
    ):
        """
        Trainer'ı başlatır.
        
        Args:
            model: Eğitilecek model
            train_loader: Eğitim veri yükleyicisi
            val_loader: Validasyon veri yükleyicisi
            device: Cihaz ('cpu' veya 'cuda')
            learning_rate: Öğrenme oranı
            num_epochs: Epoch sayısı
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss ve optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )
        
        # Metrikleri saklamak için
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Tek bir epoch eğitim yapar.
        
        Returns:
            Epoch metrikleri (loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrikler
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self) -> Dict[str, float]:
        """
        Validasyon yapar.
        
        Returns:
            Validasyon metrikleri (loss, accuracy, precision, recall, f1)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, save_path: Optional[str] = None) -> Dict[str, list]:
        """
        Tam eğitim döngüsünü çalıştırır.
        
        Args:
            save_path: En iyi modeli kaydetmek için yol (opsiyonel)
            
        Returns:
            Eğitim geçmişi
        """
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 50)
            
            # Eğitim
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validasyon
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Learning rate scheduler
            self.scheduler.step()
            
            # Sonuçları yazdır
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # En iyi modeli kaydet
            if val_metrics['accuracy'] > best_val_acc and save_path:
                best_val_acc = val_metrics['accuracy']
                save_model(self.model, save_path)
                print(f"En iyi model kaydedildi! (Val Acc: {best_val_acc:.4f})")
        
        return self.history


def prepare_data_loaders(
    data_dir: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple:
    """
    Veri yükleyicileri hazırlar.
    
    Args:
        data_dir: Veri dizini (train ve test klasörlerini içermeli)
        image_size: Görüntü boyutu
        batch_size: Batch boyutu
        num_workers: DataLoader worker sayısı
        
    Returns:
        (train_loader, val_loader, class_names) tuple'ı
    """
    preprocessor = ImagePreprocessor(image_size=image_size)
    
    # Train dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=preprocessor.train_transform
    )
    
    # Test/Val dataset
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=preprocessor.transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, class_names

