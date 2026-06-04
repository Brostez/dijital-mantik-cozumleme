from ultralytics import YOLO

# 1. Modeli Yükle
model = YOLO("yolov8n.pt")

# 2. Eğitimi Başlat
if __name__ == '__main__':
    print("Eğitim başlıyor... Bu işlem bilgisayarının hızına göre biraz zaman alabilir.")
    results = model.train(
        data="data.yaml",
        epochs=200,
        imgsz=640,
        plots=True # Eğitim grafiklerini (Hocanın istediği) otomatik çizer
    )
    print("Eğitim tamamlandı! En iyi model 'runs/detect/train/weights/best.pt' yoluna kaydedildi.")