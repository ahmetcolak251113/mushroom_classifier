import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
tf.get_logger().setLevel(logging.ERROR)

try:
    from documents import get_data_paths
except ImportError:
    print("HATA: documents.py dosyası 'src' klasöründe bulunamadı.")
    exit()
print("TensorFlow Sürümü:", tf.__version__)
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "Mushrooms")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT_DIR, "results")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mushroom_classifier_model.keras")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
NUM_CLASSES = 9


print(f"\n1. Adım: Veri yolları '{DATA_PATH}' konumundan okunuyor...")
image_paths, labels, class_names = get_data_paths(DATA_PATH)

if not image_paths:
    print("HATA: Hiç resim yolu bulunamadı. Lütfen documents.py testini kontrol edin.")
    exit()

print(f"Başarılı. {len(image_paths)} resim ve {len(class_names)} sınıf bulundu.")
print(f"Sınıflar: {class_names}")

print("\n2. Adım: Etiketler (Labels) sayısallaştırılıyor...")
encoder = LabelEncoder()
numeric_labels = encoder.fit_transform(labels)

np.save(os.path.join(MODEL_SAVE_DIR, 'class_names.npy'), encoder.classes_)
print(f"Sınıf isimleri {os.path.join(MODEL_SAVE_DIR, 'class_names.npy')} dosyasına kaydedildi.")

print(f"\n3. Adım: Veri seti {1 - VALIDATION_SPLIT:.0%} Eğitim / {VALIDATION_SPLIT:.0%} Doğrulama olarak ayrılıyor...")
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths,
    numeric_labels,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=numeric_labels
)
print(f"Eğitim verisi boyutu: {len(X_train_paths)}")
print(f"Doğrulama verisi boyutu: {len(X_val_paths)}")

def load_and_preprocess_image(path, label):
    try:
        # Resmi oku ve decode et
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=IMG_CHANNELS, expand_animations=False)

        # Şekli ayarla
        image.set_shape([None, None, IMG_CHANNELS])

        # Yeniden boyutlandır
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

        return image, label

    except Exception as e:
        tf.print(f"UYARI: Okunamayan resim atlanıyor - {path}, Hata: {e}")
        return None, None

def configure_dataset(paths, labels, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x, y: x is not None)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_ds = configure_dataset(X_train_paths, y_train, BATCH_SIZE, shuffle=True)
val_ds = configure_dataset(X_val_paths, y_val, BATCH_SIZE)

print("\n5. Adım: CNN modeli oluşturuluyor")
# CNN modeli
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    Rescaling(1. / 255),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

print(f"Eğitim {EPOCHS} epoch için başlatılıyor...")
print("İlerleme:")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping],
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print(f"\nEğitim tamamlandı. Model '{MODEL_SAVE_PATH}' adresine kaydedildi.")

print("\nFinal Model Değerlendirmesi:")
train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

print(f"Eğitim Doğruluğu: {train_accuracy:.4f}")
print(f"Eğitim Kaybı: {train_loss:.4f}")
print(f"Doğrulama Doğruluğu: {val_accuracy:.4f}")
print(f"Doğrulama Kaybı: {val_loss:.4f}")

try:
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Eğitim Doğruluğu', linewidth=2)
    plt.plot(epochs_range, val_acc, 'r-', label='Doğrulama Doğruluğu', linewidth=2)
    plt.title('Model Doğruluğu', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Eğitim Kaybı', linewidth=2)
    plt.plot(epochs_range, val_loss, 'r-', label='Doğrulama Kaybı', linewidth=2)
    plt.title('Model Kaybı', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nEğitim grafikleri '{plot_path}' adresine kaydedildi.")


except ImportError:
    print("\nGrafik oluşturmak için 'matplotlib' kütüphanesi gerekli.")
    print("Kurulum için: pip install matplotlib")

except Exception as e:
    print(f"\nGrafik oluşturulurken hata oluştu: {e}")


print("\n" + "=" * 60)
print("EĞİTİM ÖZETİ")
print("=" * 60)
print(f"Toplam Eğitim Örnekleri: {len(X_train_paths)}")
print(f"Toplam Doğrulama Örnekleri: {len(X_val_paths)}")
print(f"Toplam Sınıf Sayısı: {NUM_CLASSES}")
print(f"Kullanılan Sınıflar: {class_names}")
print(f"Toplam Epoch Sayısı: {len(history.history['loss'])}")
print(f"Son Eğitim Doğruluğu: {history.history['accuracy'][-1]:.4f}")
print(f"Son Doğrulama Doğruluğu: {history.history['val_accuracy'][-1]:.4f}")
print("=" * 60)