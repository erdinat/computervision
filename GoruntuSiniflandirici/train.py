"""
Model eğitim scripti
"""

import argparse
import torch
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer import Trainer, prepare_data_loaders
from src.model_loader import create_model


def main():
    parser = argparse.ArgumentParser(description='Görüntü Sınıflandırıcı Model Eğitimi')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Veri dizini yolu (train ve test klasörlerini içermeli)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Epoch sayısı (varsayılan: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch boyutu (varsayılan: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Öğrenme oranı (varsayılan: 0.001)'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='models/best_model.pth',
        help='Model kayıt yolu (varsayılan: models/best_model.pth)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Görüntü boyutu (varsayılan: 224 224)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader worker sayısı (varsayılan: 4)'
    )
    
    args = parser.parse_args()
    
    # Cihaz kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Kullanılan cihaz: {device}")
    
    # Veri yükleyicileri hazırla
    print("\nVeri yükleyicileri hazırlanıyor...")
    train_loader, val_loader, class_names = prepare_data_loaders(
        data_dir=args.data_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Sınıf sayısı: {len(class_names)}")
    print(f"Sınıflar: {class_names}")
    print(f"Eğitim örnekleri: {len(train_loader.dataset)}")
    print(f"Validasyon örnekleri: {len(val_loader.dataset)}")
    
    # Model oluştur
    print("\nModel oluşturuluyor...")
    model = create_model(num_classes=len(class_names), pretrained=True)
    print(f"Model parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )
    
    # Eğitimi başlat
    print("\nEğitim başlatılıyor...")
    history = trainer.train(save_path=args.model_save_path)
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("Eğitim tamamlandı!")
    print("="*50)
    print(f"En yüksek validasyon accuracy: {max(history['val_acc']):.4f}")
    print(f"Model kaydedildi: {args.model_save_path}")
    
    # Sınıf isimlerini kaydet (opsiyonel)
    class_names_path = os.path.join(
        os.path.dirname(args.model_save_path),
        'class_names.txt'
    )
    with open(class_names_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Sınıf isimleri kaydedildi: {class_names_path}")

if __name__ == "__main__":
    main()
