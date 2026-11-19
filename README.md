# SafeVision 👁️🛡️

Sistema de monitoreo, videovigilancia y seguridad industrial en tiempo real basado en visión por computadora (YOLO) con control y automatización de hardware mediante microcontrolador ESP32.

---

## 🚀 Características Principales

- **Visión Artificial con YOLOv11:** Detección de personas, equipo de protección personal (EPP/lentes) y objetos en tiempo real con ejecución en hilo independiente para cero congelamiento de interfaz.
- **Gestión Avanzada de Cámaras:** Soporte para webcams USB, cámaras IP (RTSP/HTTP), perfiles de conexión predefinidos y escaneo automático de dispositivos en red local.
- **Zonas de Seguridad y Cruce de Líneas:** Delimitación de polígonos interactivos de zonas seguras/restringidas y detección de cruce de líneas perimetrales.
- **Integración con Hardware (ESP32):** Comunicación serie/USB o red con microcontrolador ESP32 para activación de relés, torretas de alarma, sirenas o paros de emergencia.
- **Interfaz Gráfica Profesional:** GUI construida en Python/Tkinter para calibración de zonas en vivo, control de estados de relés, visualización multipantalla y registro de eventos.

---

## 📁 Estructura del Proyecto

```text
├── camera_manager.py              # Gestión, reconexión y streaming de cámaras (USB e IP)
├── esp32_controller.py            # Controlador de comunicación y comandos con ESP32
├── safevision_gui.py              # Interfaz gráfica principal del sistema
├── line_crossing.py               # Lógica matemática de cruce de líneas y zonas
├── safevision_config.json         # Configuración persistente del sistema y zonas
├── yolo11n.pt                     # Modelo base YOLOv11
├── esp32_reles_safevision/        # Código fuente Arduino/C++ para el ESP32
│   └── esp32_reles_safevision.ino
├── dataset_lentes/                # Dataset y scripts de entrenamiento de EPP
├── docs / diagramas:
│   ├── Diagrama_Bloques_Hardware.pdf
│   └── Modelo_ER_Seguridad_Industrial.pdf
└── .gitignore                     # Filtro de archivos innecesarios para Git
```

---

## 🛠️ Requisitos e Instalación

### Requisitos previos
- Python 3.9 o superior
- Arduino IDE (para flashear el ESP32)

### Instalación de dependencias
```bash
pip install ultralytics opencv-python pillow pyserial numpy
```

### Ejecución
```bash
python safevision_gui.py
```

---

## 📄 Licencia
Proyecto desarrollado para seguridad y automatización industrial.
