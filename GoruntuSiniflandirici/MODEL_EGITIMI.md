# Model Eğitimi Rehberi

## Hızlı Başlangıç

### 1. Veri Seti İndirme

Aşağıdaki veri setlerinden birini seçin ve indirin:

#### CIFAR-10 (Önerilen - Küçük ve Hızlı)
- **İndirme:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Boyut:** ~170 MB
- **Sınıflar:** 10 sınıf (uçak, otomobil, kuş, kedi, geyik, köpek, kurbağa, at, gemi, kamyon)

#### Animals-10
- **İndirme:** https://www.kaggle.com/datasets/alessiocorrado99/animals10
- **Sınıflar:** 10 hayvan sınıfı

#### Fruits 360
- **İndirme:** https://www.kaggle.com/datasets/moltean/fruits
- **Sınıflar:** 130+ meyve sınıfı

### 2. Veri Yapısını Oluşturma

Veri setini şu yapıda düzenleyin:

```
data/
├── train/
│   ├── airplane/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── automobile/
│   ├── bird/
│   └── ...
└── test/
    ├── airplane/
    ├── automobile/
    └── ...
```

**Önemli:** Her sınıf için ayrı klasör oluşturun ve görüntüleri ilgili klasöre yerleştirin.

### 3. Model Eğitimi

Terminal'de şu komutu çalıştırın:

```bash
cd GoruntuSiniflandirici
python train.py --data_dir data --epochs 10 --batch_size 32
```

**Parametreler:**
- `--data_dir`: Veri dizini (varsayılan: `data`)
- `--epochs`: Eğitim turu sayısı (varsayılan: 10)
- `--batch_size`: Her seferde işlenecek görüntü sayısı (varsayılan: 32)
- `--learning_rate`: Öğrenme oranı (varsayılan: 0.001)

**Örnek:**
```bash
# Hızlı test için (5 epoch)
python train.py --data_dir data --epochs 5 --batch_size 16

# Daha iyi sonuç için (20 epoch)
python train.py --data_dir data --epochs 20 --batch_size 32
```

### 4. Eğitim Sonrası

Eğitim tamamlandığında:
- Model `models/best_model.pth` olarak kaydedilir
- Sınıf isimleri `models/class_names.txt` olarak kaydedilir
- Web arayüzünde artık tahmin yapabilirsiniz!

### 5. Örnek Eğitim Sonuçları

**Animals-10 (5 sınıf) Eğitim Sonuçları:**

```
Veri Seti: Animals-10
Sınıflar: fil, kelebek, kedi, koyun, örümcek
Eğitim örnekleri: 1,573
Test örnekleri: 397

Epoch 1/10:
- Train Acc: 74.51% | Val Acc: 57.18%
- Precision: 67.18% | Recall: 57.18% | F1: 53.95%

Epoch 2/10:
- Train Acc: 79.02% | Val Acc: 86.15% ✅
- Precision: 87.73% | Recall: 86.15% | F1: 86.03%
- EN İYİ MODEL KAYDEDİLDİ!

Sonuç: Model başarıyla eğitildi ve %86.15 validation accuracy elde etti!
```

**Not:** Daha fazla epoch ile (10-20) %90+ accuracy elde edilebilir.

## Sorun Giderme

### "Veri bulunamadı" hatası
- `data/train/` ve `data/test/` klasörlerinin var olduğundan emin olun
- Her klasörde en az bir görüntü olduğundan emin olun

### "CUDA out of memory" hatası
- `--batch_size` değerini küçültün (örn: 16 veya 8)
- CPU kullanımı için otomatik olarak CPU'ya geçer

### Eğitim çok yavaş
- `--batch_size` değerini artırın
- `--num_workers` parametresini ayarlayın (varsayılan: 4)

## İpuçları

1. **İlk test için:** Küçük bir veri setiyle başlayın (örn: 2-3 sınıf, her sınıftan 50-100 görüntü)
2. **Daha iyi sonuçlar için:** Daha fazla epoch kullanın (20-30)
3. **Hızlı test için:** 5 epoch yeterli olabilir
4. **GPU varsa:** Otomatik olarak kullanılır, çok daha hızlı eğitim yapar

