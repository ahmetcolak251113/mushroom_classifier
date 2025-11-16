import os
import tensorflow as tf
from documents import get_data_paths
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "Mushrooms")
print("Veri temizleme script'i başlatıldı (TensorFlow ile).")
print("Bozuk resimler taranıyor. Bu işlem biraz zaman alabilir...\n")
image_paths, _, _ = get_data_paths(DATA_PATH)

if not image_paths:
    print("Hiç resim bulunamadı. Script durduruldu.")
    exit()

corrupt_file_count = 0
total_files_checked = len(image_paths)

for i, path in enumerate(image_paths):
    if (i + 1) % 100 == 0:
        print(f"  Kontrol ediliyor: {i + 1} / {total_files_checked}", end='\r')
    try:
        image_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)

    except tf.errors.InvalidArgumentError as e:
        print(f"\n!!! BOZUK DOSYA BULUNDU (TensorFlow): {path}")
        try:
            os.remove(path)
            print(f"    -> DOSYA SİLİNDİ.")
            corrupt_file_count += 1
        except OSError as oe:
            print(f"    -> HATA: Dosya silinemedi. Hata: {oe}")
    except Exception as e:
        print(f"\n!!! BİLİNMEYEN HATA: {path}")
        print(f"    Hata: {e}")
print("\n" + "=" * 50)
print("Tarama tamamlandı.")
print(f"Kontrol edilen toplam dosya sayısı: {total_files_checked}")
print(f"Bulunan ve silinen bozuk dosya sayısı: {corrupt_file_count}")

if corrupt_file_count == 0:
    print("Veri setiniz temiz görünüyor!")
else:
    print(f"Veri setinizden {corrupt_file_count} adet bozuk dosya temizlendi.")
    print("Artık 'train.py' script'ini çalıştırmayı deneyebilirsiniz.")