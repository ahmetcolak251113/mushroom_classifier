import os


def get_data_paths(dataset_base_path):
    image_paths = []
    labels = []
    IGNORE_LIST = ['Mushrooms', '.DS_Store']
    try:
        class_names = sorted([
            d for d in os.listdir(dataset_base_path)
            if os.path.isdir(os.path.join(dataset_base_path, d)) and d not in IGNORE_LIST
        ])
    except FileNotFoundError:
        print(f"HATA: Belirtilen yolda klasör bulunamadı: {dataset_base_path}")
        print("Lütfen 'Mushrooms' klasörünü 'data/raw/' içine taşıdığınızdan emin olun.")
        return None, None, None

    if not class_names:
        print(f"HATA: {dataset_base_path} içinde geçerli hiçbir sınıf klasörü bulunamadı.")
        print("Klasör yapınızı kontrol edin.")
        return None, None, None

    print(f"Toplam {len(class_names)} geçerli sınıf bulundu:")
    print(f"{class_names}\n")

    for class_name in class_names:
        class_dir = os.path.join(dataset_base_path, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(class_dir, file_name)
                image_paths.append(image_path)
                labels.append(class_name)  # Etiket olarak klasör adını (cins adını) ekle

    print(f"Toplam {len(image_paths)} resim dosyası bulundu.")

    return image_paths, labels, class_names
if __name__ == "__main__":
    PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "Mushrooms")
    print(f"Veri seti şu konumda aranıyor: {DATA_PATH}\n")
    paths, labels, classes = get_data_paths(DATA_PATH)
    if paths:
        print("\n--- TEST BAŞARILI ---")
        print(f"İlk 5 resim yolu: {paths[:5]}")
        print(f"İlk 5 etiket: {labels[:5]}")
    else:
        print("\n--- TEST BAŞARISIZ ---")
        print("Veri yolları okunamadı. Lütfen yukarıdaki HATA mesajlarını kontrol edin.")