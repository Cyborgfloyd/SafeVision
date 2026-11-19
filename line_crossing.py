import time
import cv2
from ultralytics import YOLO
from safevision_gui import (
    SmartCamaraIP,
    cargar_config,
    guardar_config,
    normalizar_entrada_camara,
    emitir_pitido,
)

# ==========================================================
# CONFIGURACIÓN (Sincronizada con safevision_config.json)
# ==========================================================
cfg = cargar_config()
URL_CAMARA = cfg.get("camera_url", "http://172.16.1.18:8080/video")
CONFIANZA_MINIMA = cfg.get("confidence", 50.0) / 100.0
ALERTA_SONORA = cfg.get("sound_alarm", True)

model = YOLO("yolo11n.pt")
camara = SmartCamaraIP(URL_CAMARA)

linea_x_pct = float(cfg.get("line_x", 0.50))
arrastrando = False
ancho_actual = 640


def mover_linea_mouse(event, x, y, flags, param):
    global linea_x_pct, arrastrando
    if event == cv2.EVENT_LBUTTONDOWN:
        arrastrando = True
        linea_x_pct = max(0.05, min(0.95, x / max(ancho_actual, 1)))
    elif event == cv2.EVENT_MOUSEMOVE and arrastrando:
        linea_x_pct = max(0.05, min(0.95, x / max(ancho_actual, 1)))
    elif event == cv2.EVENT_LBUTTONUP:
        arrastrando = False
        cfg["line_x"] = round(linea_x_pct, 3)
        guardar_config(cfg)


ventana = "Detector de Personas - Camara Celular"
cv2.namedWindow(ventana)
cv2.setMouseCallback(ventana, mover_linea_mouse)

print("--- DETECTOR DE PERSONAS (CÁMARA DEL CELULAR VÍA HOTSPOT) ---")
print(f"Fuente actual: {normalizar_entrada_camara(URL_CAMARA)}")
print("- Arrastra con el mouse para mover la línea roja vertical.")
print("- Presiona 'i' en la ventana para escribir una nueva IP de cámara.")
print("- Presiona 'w' para cambiar a la Webcam de la PC (0).")
print("- Presiona 'q' para salir.")

estado_anterior = None
ultimo_pitido = 0.0
tiempo_anterior = time.time()

while True:
    frame, conectada, estado_txt, url_norm = camara.leer()

    if frame is None:
        import numpy as np
        frame = np.zeros((480, 800, 3), dtype=np.uint8)
        cv2.putText(frame, f"Estado: {estado_txt}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, "Presiona 'i' en esta ventana y mira la consola para cambiar la IP", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "O ejecuta: python safevision_gui.py para usar la interfaz Figma", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1)

    alto, ancho_actual, _ = frame.shape
    linea_x = int(ancho_actual * linea_x_pct)

    cv2.line(frame, (linea_x, 0), (linea_x, alto), (0, 0, 255), 4)
    cv2.putText(frame, "ZONA SEGURA (IZQ)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "ZONA RESTRINGIDA (DER)", (linea_x + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    en_zona_restringida = False
    personas = 0

    if conectada:
        results = model(frame, classes=[0], conf=CONFIANZA_MINIMA, verbose=False)
        for r in results:
            for box in r.boxes:
                personas += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confianza = float(box.conf[0])
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                cv2.circle(frame, (centro_x, centro_y), 6, (255, 255, 0), -1)

                if centro_x > linea_x:
                    en_zona_restringida = True
                    color, texto = (0, 0, 255), f"INTRUSION {confianza:.0%}"
                else:
                    color, texto = (0, 255, 0), f"SEGURO {confianza:.0%}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, texto, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if en_zona_restringida != estado_anterior:
        if en_zona_restringida:
            print("[ALERTA] ¡Intrusión detectada en la zona restringida derecha!")
        else:
            print("[ESTABLE] Zona restringida despejada.")
        estado_anterior = en_zona_restringida

    if en_zona_restringida:
        cv2.rectangle(frame, (0, 0), (ancho_actual - 1, alto - 1), (0, 0, 255), 10)
        if ALERTA_SONORA and time.time() - ultimo_pitido > 1:
            emitir_pitido()
            ultimo_pitido = time.time()

    ahora = time.time()
    fps = 1 / max(ahora - tiempo_anterior, 1e-6)
    tiempo_anterior = ahora
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Personas: {personas} | {estado_txt[:38]}",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow(ventana, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("w"):
        camara.cambiar_fuente("0")
    elif key == ord("i"):
        nueva_ip = input("\nEscribe la nueva IP:Puerto de la app del celular (ej. 172.16.1.18:8080): ").strip()
        if nueva_ip:
            cfg["camera_url"] = normalizar_entrada_camara(nueva_ip)
            guardar_config(cfg)
            camara.cambiar_fuente(nueva_ip)

camara.detener()
cv2.destroyAllWindows()
