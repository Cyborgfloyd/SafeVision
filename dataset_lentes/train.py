from ultralytics import YOLO

def main():
    # 1. Cargar el modelo base YOLO11 Nano (ideal para entrenar rápido en laptops)
    model = YOLO("yolo11n.pt")

    print("--- INICIANDO ENTRENAMIENTO DE LENTES ---")
    
    # 2. Ejecutar el entrenamiento optimizado para tu RTX 3050
    results = model.train(
        data="data.yaml",    # Archivo de configuración que creamos
        epochs=50,           # Cantidad de veces que repasará las fotos (50 es un buen inicio)
        imgsz=640,           # Resolución de las imágenes
        batch=4,             # Lote pequeño (batch=4) para cuidar la memoria VRAM de tu 3050
        device=0             # Forzar el uso de la tarjeta gráfica NVIDIA
    )

    print("¡Entrenamiento finalizado con éxito!")
    print("Tu modelo entrenado se guardó en: runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()