import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class MushroomClassifier:
    def __init__(self):
        self.PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_PATH = os.path.join(self.PROJECT_ROOT_DIR, "models", "mushroom_classifier_model.keras")
        self.CLASS_NAMES_PATH = os.path.join(self.PROJECT_ROOT_DIR, "models", "class_names.npy")
        self.model = None
        self.class_names = []
        self.load_model_and_classes()
        self.camera = None
        self.img_size = (224, 224)

    def load_model_and_classes(self):
        try:
            print("Model yükleniyor...")
            self.model = load_model(self.MODEL_PATH)
            self.class_names = np.load(self.CLASS_NAMES_PATH)
            print(f"Model başarıyla yüklendi! {len(self.class_names)} sınıf tanınıyor.")
            print(f"Sınıflar: {list(self.class_names)}")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            return False
        return True

    def preprocess_image(self, image):
        image_resized = cv2.resize(image, self.img_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch

    def predict_image(self, image):
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            return predicted_class, confidence, predictions[0]

        except Exception as e:
            print(f"Tahmin yapılırken hata oluştu: {e}")
            return None, 0, None

    def draw_prediction_info(self, image, prediction, confidence, top_predictions=None):
        cv2.rectangle(image, (10, 10), (400, 150), (0, 0, 0), -1)
        text = f"{prediction}: %{confidence:.1f}"
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if top_predictions is not None:
            top_indices = np.argsort(top_predictions)[-3:][::-1]
            y_offset = 70
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx]
                conf = top_predictions[idx] * 100
                text = f"{i + 1}. {class_name}: %{conf:.1f}"
                cv2.putText(image, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25

        cv2.putText(image, "Q: Cikis  S: Kaydet", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return image

    def save_image(self, image, prediction, confidence):
        try:
            test_images_dir = os.path.join(self.PROJECT_ROOT_DIR, "data", "test_images")
            os.makedirs(test_images_dir, exist_ok=True)
            filename = f"{prediction}_{confidence:.2f}.jpg"
            filepath = os.path.join(test_images_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"Görüntü kaydedildi: {filepath}")
        except Exception as e:
            print(f"Görüntü kaydedilirken hata oluştu: {e}")

    def run_camera(self):
        if self.model is None:
            print("Model yüklenemedi!")
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Kamera açılamadı!")
            return
        print("\nKamera başlatıldı. Tahminler başlıyor...")
        print("Talimatlar:")
        print("- Q: Çıkış")
        print("- S: Görüntüyü kaydet")
        print("- Boşluk: Tahmin yap")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Görüntü alınamadı!")
                break
            frame = cv2.flip(frame, 1)
            prediction, confidence, all_predictions = self.predict_image(frame)
            if prediction:
                processed_frame = self.draw_prediction_info(frame.copy(), prediction, confidence, all_predictions)
            else:
                processed_frame = frame
            cv2.imshow('Mantar Tanima Sistemi', processed_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s') and prediction:
                self.save_image(frame, prediction, confidence)
            elif key == ord(' '):
                # Boşluk tuşuna basıldığında tahmin yap
                if prediction:
                    print(f"Tahmin: {prediction}, Güven: %{confidence:.2f}")
        self.camera.release()
        cv2.destroyAllWindows()
        print("Kamera kapatıldı.")

def main():
    print("Mantar Tanıma Sistemine Hoşgeldiniz!")
    classifier = MushroomClassifier()
    if classifier.model is not None:
        classifier.run_camera()
    else:
        print("Sistem başlatılamadı!")
if __name__ == "__main__":
    main()