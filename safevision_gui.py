import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import datetime
import time
import threading
import json
import csv
import os
import uuid

from camera_manager import (
    SmartCamaraIP,
    CameraPool,
    EscanerRedYWebcams,
    CAMERA_PROFILES,
    construir_url_camara,
)
from esp32_controller import ControladorESP32Reles

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

CONFIG_FILE = "safevision_config.json"

ZONE_COLORS_BGR = [
    (68, 68, 239),    # Rojo industrial (#EF4444)
    (11, 158, 245),   # Ámbar (#F59E0B)
    (246, 130, 59),   # Azul (#3B82F6)
    (247, 85, 168),   # Morado (#A855F7)
    (182, 182, 20),   # Cyan (#14B8A6)
]

ZONE_COLORS_HEX = [
    "#EF4444",
    "#F59E0B",
    "#3B82F6",
    "#A855F7",
    "#14B8A6",
]


# =====================================================================
# MOTOR DE INTELIGENCIA ARTIFICIAL EN HILO DEDICADO (CERO BLOQUEO UI)
# =====================================================================
class MotorYOLO:
    """
    Ejecuta YOLO11 en un hilo separado para que la interfaz gráfica y las cámaras
    corran a máxima fluidez sin esperar los ~25ms de la red neuronal.
    """
    def __init__(self, model_path="yolo11n.pt", imgsz=320):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_min = 0.30
        self.stopped = False

        self._input_lock = threading.Lock()
        self._input_frame = None
        self._input_id = -1
        self._input_source = "main"
        self._event = threading.Event()

        self._output_lock = threading.Lock()
        self.detections_by_source = {"main": []}
        self.infer_ms = 0.0

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def enviar_frame(self, frame, frame_id, source_key="main"):
        with self._input_lock:
            if frame_id != self._input_id or source_key != self._input_source:
                self._input_frame = frame
                self._input_id = frame_id
                self._input_source = source_key
                self._event.set()

    def obtener_detecciones(self, source_key="main"):
        with self._output_lock:
            return list(self.detections_by_source.get(source_key, [])), self.infer_ms

    def _loop(self):
        last_key = None
        while not self.stopped:
            self._event.wait(timeout=0.05)
            self._event.clear()
            if self.stopped:
                break

            with self._input_lock:
                if self._input_frame is None:
                    continue
                cur_key = (self._input_source, self._input_id)
                if cur_key == last_key:
                    continue
                frm = self._input_frame
                src = self._input_source
                last_key = cur_key

            t0 = time.perf_counter()
            try:
                results = self.model(
                    frm,
                    classes=[0],
                    imgsz=self.imgsz,
                    conf=self.conf_min,
                    verbose=False,
                    half=False
                )
                dets = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        for (x1, y1, x2, y2), c in zip(xyxy, confs):
                            dets.append((int(x1), int(y1), int(x2), int(y2), float(c)))
                ms = (time.perf_counter() - t0) * 1000.0
                with self._output_lock:
                    self.detections_by_source[src] = dets
                    self.infer_ms = ms
            except Exception:
                pass

    def stop(self):
        self.stopped = True
        self._event.set()


# =====================================================================
# INTERFAZ PRINCIPAL SAFEVISION (DISEÑO FIGMA + GESTOR CAMARAS + MONITOR 4X)
# =====================================================================
class SafeVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Monitoreo Industrial")
        self.root.geometry("1460x930")
        self.root.minsize(1240, 780)
        self.root.configure(bg="#090D16")

        # Paleta de colores original de Figma
        self.C_BG_MAIN    = "#090D16"
        self.C_BG_SIDEBAR = "#111827"
        self.C_BG_CARD    = "#111827"
        self.C_BG_CARD2   = "#1E293B"
        self.C_BORDER     = "#1F2937"
        self.C_BLUE       = "#2563EB"
        self.C_GREEN      = "#10B981"
        self.C_RED        = "#EF4444"
        self.C_AMBER      = "#F59E0B"
        self.C_TEXT       = "#F9FAFB"
        self.C_MUTED      = "#9CA3AF"

        # Configuración persistente
        self.config = self._cargar_config()

        # Estado operativo
        self.sistema_armado = True
        self.paro_emergencia = False
        self.vista_actual = "monitoreo"

        # Estado de edición interactiva de polígonos y líneas
        self.selected_zone_idx = 0
        self.modo_agregar_puntos = False
        self.dragging_mode = None
        self.drag_zone_idx = 0
        self.drag_vertex_idx = 0
        self.drag_start_norm = (0.0, 0.0)
        self.drag_orig_points = []
        self.hover_info = None
        self.zone_clipboard = None  # Portapapeles para Copiar / Pegar zonas

        # Dimensiones del video principal
        self.canvas_w = 880
        self.canvas_h = 540
        self.vid_x0 = 0
        self.vid_y0 = 0
        self.vid_w = 880
        self.vid_h = 540

        # Estado del Gestor de Cámaras (CRUD)
        self.editing_camera_id = None
        self.ultimas_camaras_detectadas = []

        # Estado del Monitor de 4 Cámaras (2x2)
        self.quad_ai_enabled = False
        self.quad_rr_index = 0
        self.quad_slots_ui = []  # Guardará referencias a los 4 paneles

        # Historial y contadores
        self.bitacora_eventos = []
        self.alertas_recientes = []
        self.ultimo_evento_ts = 0.0
        self.ultimo_beep_ts = 0.0
        self.total_intrusiones = 0
        self.fps_actual = 0.0
        self._fps_t0 = time.time()
        self._fps_frames = 0
        self._last_frame_id = -1

        # Inicializar motor YOLO en segundo plano
        print("Cargando motor YOLO11 en hilo dedicado...")
        self.yolo_engine = MotorYOLO("yolo11n.pt", imgsz=int(self.config.get("yolo_imgsz", 320)))

        # Pool de Cámaras compartidas (evita bloqueos y duplicación de recursos)
        self.modo_prueba_pc = False
        self.ultima_ip_real = self.config["camera_url"] if self.config["camera_url"] != "0" else "http://10.66.172.56:8080/video"
        self.cam_pool = CameraPool()
        self.cam = self._conectar_camara_principal_pool(self.config["camera_url"])

        # Controlador de 4 Relés ESP32 (USB Serial COM + Wi-Fi simultáneo)
        self.esp32 = ControladorESP32Reles(
            mode=self.config.get("esp32_mode", "ambos"),
            serial_port=self.config.get("esp32_port", "AUTO"),
            baudrate=int(self.config.get("esp32_baud", 115200)),
            wifi_ip=self.config.get("esp32_wifi_ip", ""),
            active_low=bool(self.config.get("esp32_active_low", True)),
        )

        # Construir interfaz completa (sin encender cámaras secundarias hasta abrir Monitor 4x)
        self._construir_ui()

        self._registrar_evento(
            "INFO",
            "Sistema Iniciado",
            f"Sistema listo. {len(self.config['saved_cameras'])} cámara(s) guardada(s), {len(self.config['geofences'])} geocerca(s). Control ESP32 4 Relés activo."
        )

        # Bucle de refresco visual rápido
        self._actualizar_loop()

    # -----------------------------------------------------------------
    # GESTIÓN DE CONFIGURACIÓN, GEOCERCAS Y CÁMARAS GUARDADAS
    # -----------------------------------------------------------------
    def _cargar_config(self):
        defaults = {
            "camera_url": "http://10.66.172.56:8080/video",
            "plc_ip": "192.168.1.100",
            "plc_port": "502",
            "esp32_mode": "ambos",
            "esp32_port": "AUTO",
            "esp32_baud": 115200,
            "esp32_wifi_ip": "",
            "esp32_active_low": True,
            "cell_name": "Celda Activa: Torno CNC 01",
            "zone_mode": "geocerca_poly",
            "invert_side": False,
            "line_x": 0.50,
            "line_y": 0.50,
            "rect_x1": 0.50,
            "rect_y1": 0.25,
            "rect_x2": 0.92,
            "rect_y2": 0.90,
            "geofences": [
                {
                    "name": "Zona 1",
                    "action": "alarma",
                    "inverted": False,
                    "points": [
                        [0.10, 0.20],
                        [0.48, 0.20],
                        [0.48, 0.88],
                        [0.10, 0.88]
                    ]
                },
                {
                    "name": "Zona 2",
                    "action": "corte",
                    "inverted": False,
                    "points": [
                        [0.54, 0.20],
                        [0.92, 0.20],
                        [0.92, 0.88],
                        [0.54, 0.88]
                    ]
                }
            ],
            "saved_cameras": [
                {
                    "id": "cam_hotspot_1",
                    "name": "Celular Hotspot (IP Webcam)",
                    "type": "http",
                    "host": "192.168.137.20",
                    "port": "8080",
                    "username": "",
                    "password": "",
                    "stream_path": "/video",
                    "url": "http://192.168.137.20:8080/video"
                }
            ],
            "quad_cameras": ["-- Desactivada --", "-- Desactivada --", "-- Desactivada --", "-- Desactivada --"],
            "auto_connect_open_cams": True,
            "detect_point": "pies",
            "max_width": 640,
            "yolo_imgsz": 320,
            "confidence": 65.0,
            "require_epp": False,
            "sound_alarm": True
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if "geofences" not in saved or not saved["geofences"]:
                    rx1 = float(saved.get("rect_x1", 0.50))
                    ry1 = float(saved.get("rect_y1", 0.25))
                    rx2 = float(saved.get("rect_x2", 0.92))
                    ry2 = float(saved.get("rect_y2", 0.90))
                    saved["geofences"] = [{
                        "name": "Zona 1",
                        "action": "alarma",
                        "points": [[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]]
                    }]
                if saved.get("zone_mode") == "geocerca_rect":
                    saved["zone_mode"] = "geocerca_poly"
                defaults.update(saved)

                # Forzar modo 100% USB Serial para máxima estabilidad con la ESP32
                defaults["esp32_mode"] = "usb"

                # Asegurar que cada geocerca tenga su propiedad 'action':
                #   - "alarma": Acción 1 -> Encender Alarma (Relé 1)
                #   - "corte":  Acción 2 -> Cortar Corriente (Relé 2 + Alarma)
                #   - "seguro": Zona Segura (Permitida)
                for idx_gz, gz in enumerate(defaults.get("geofences", [])):
                    if "name" not in gz:
                        gz["name"] = f"Zona {idx_gz + 1}"
                    if "inverted" not in gz:
                        gz["inverted"] = False
                    if "action" not in gz or gz["action"] not in ("alarma", "corte", "seguro"):
                        if gz.get("inverted", False):
                            gz["action"] = "seguro"
                        else:
                            gz["action"] = "alarma" if idx_gz == 0 else "corte"
                    gz["inverted"] = (gz["action"] == "seguro")

                # NUNCA encender la cámara de la laptop ("0") al iniciar automáticamente
                if str(defaults.get("camera_url", "")).strip() == "0":
                    ip_fallback = "http://10.66.172.56:8080/video"
                    for c in defaults.get("saved_cameras", []):
                        if str(c.get("url", "")).strip() not in ("0", ""):
                            ip_fallback = c["url"]
                            break
                    defaults["camera_url"] = ip_fallback

                # Eliminar cualquier referencia automática a la cámara "0" de la laptop en saved_cameras o quad_cameras
                defaults["saved_cameras"] = [
                    c for c in defaults.get("saved_cameras", [])
                    if str(c.get("url", "")).strip() != "0" and str(c.get("host", "")).strip() != "0"
                ]
                qc = [
                    ("-- Desactivada --" if str(x).strip() == "0" else x)
                    for x in defaults.get("quad_cameras", [])
                ]
                while len(qc) < 4:
                    qc.append("-- Desactivada --")
                defaults["quad_cameras"] = qc[:4]
            except Exception as e:
                print("Error leyendo config:", e)
        return defaults

    def _guardar_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("Error guardando config:", e)

    def _buscar_credenciales_guardadas(self, url_or_index):
        u_str = str(url_or_index).strip()
        # Extraer host limpio si viene como rtsp://... o http://...
        host_limpio = u_str
        if "://" in host_limpio:
            host_limpio = host_limpio.split("://", 1)[1]
        if "@" in host_limpio:
            host_limpio = host_limpio.rsplit("@", 1)[1]
        host_limpio = host_limpio.split("/")[0].split(":")[0].strip()

        for c in self.config.get("saved_cameras", []):
            c_url = str(c.get("url", "")).strip()
            c_host = str(c.get("host", "")).strip()
            if c_url == u_str or c_host == u_str or (host_limpio and c_host == host_limpio):
                return c.get("username", ""), c.get("password", "")
        return "", ""

    def _resolver_url_con_guardadas(self, raw_input):
        """
        Si el usuario escribe solo la IP (ej. 192.168.1.64) o selecciona una URL sin credenciales,
        busca si esa IP ya está en saved_cameras para usar su URL RTSP/HTTP completa y sus credenciales.
        """
        t = str(raw_input).strip()
        if not t or t.isdigit():
            return t
        host_in = t
        if "://" in host_in:
            host_in = host_in.split("://", 1)[1]
        if "@" in host_in:
            host_in = host_in.rsplit("@", 1)[1]
        host_in = host_in.split("/")[0].split(":")[0].strip()

        for c in self.config.get("saved_cameras", []):
            c_host = str(c.get("host", "")).strip()
            c_url = str(c.get("url", "")).strip()
            if c_url == t or (host_in and c_host == host_in):
                return construir_url_camara(c)
        return str(SmartCamaraIP.normalizar_url(t))

    def _conectar_camara_principal_pool(self, url_or_index):
        user, pwd = self._buscar_credenciales_guardadas(url_or_index)
        cam = self.cam_pool.obtener_o_crear(
            url_or_index,
            max_width=int(self.config.get("max_width", 640)),
            username=user,
            password=pwd
        )
        return cam

    def _limpiar_pool_inactivas(self):
        cam_actual = "0" if self.modo_prueba_pc else self.config.get("camera_url", "")
        activas = [cam_actual]
        if self.vista_actual == "multicam":
            for q_url in self.config.get("quad_cameras", []):
                if q_url and q_url not in ("-- Desactivada --", "0"):
                    activas.append(q_url)
        self.cam_pool.limpiar_no_usadas(activas)

    # -----------------------------------------------------------------
    # CONSTRUCCIÓN DE LA INTERFAZ (FIGMA + 5 VISTAS)
    # -----------------------------------------------------------------
    def _construir_ui(self):
        self.root.grid_columnconfigure(0, weight=0, minsize=255)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ==================== 1. SIDEBAR IZQUIERDO ====================
        self.sidebar = tk.Frame(
            self.root, bg=self.C_BG_SIDEBAR, width=255,
            highlightbackground=self.C_BORDER, highlightthickness=1
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Espacio superior limpio (sin emblema ni nombre fijo)
        tk.Label(
            self.sidebar, text="MENÚ DE NAVEGACIÓN",
            font=("Segoe UI", 8, "bold"), bg=self.C_BG_SIDEBAR, fg=self.C_MUTED
        ).pack(anchor="w", padx=18, pady=(16, 8))

        # Botones de navegación limpios
        self.nav_buttons = {}
        nav_items = [
            ("monitoreo",   "📹  Monitoreo en Vivo"),
            ("multicam",    "📺  Monitor 4 Cámaras"),
            ("camaras",     "📡  Gestor de Cámaras"),
            ("bitacora",    "📋  Bitácora de Eventos"),
            ("calibracion", "⚙️  Configuración"),
        ]
        for key, text in nav_items:
            btn = tk.Button(
                self.sidebar, text=text, font=("Segoe UI", 10, "bold"),
                bg=self.C_BLUE if key == "monitoreo" else self.C_BG_SIDEBAR,
                fg=self.C_TEXT if key == "monitoreo" else self.C_MUTED,
                activebackground=self.C_BLUE, activeforeground="white",
                bd=0, relief="flat", anchor="w", padx=16, pady=10, cursor="hand2",
                command=lambda k=key: self._cambiar_vista(k)
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[key] = btn

        # Guía rápida compacta
        help_box = tk.Frame(
            self.sidebar, bg="#0F172A",
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=12, pady=10
        )
        help_box.pack(fill="x", padx=12, pady=(14, 8))

        tk.Label(
            help_box, text="💡 GUÍA RÁPIDA",
            font=("Segoe UI", 8, "bold"), bg="#0F172A", fg="#60A5FA"
        ).pack(anchor="w", pady=(0, 5))

        pasos = [
            ("🟡 Arrastrar punto", "Mueve esquina de zona"),
            ("➕ Doble clic", "Agrega más esquinas"),
            ("🖱️ Clic derecho", "Menú de opciones de zona"),
            ("📡 Gestor de Cámaras", "Escanea y guarda IPs"),
            ("📺 Monitor 4 Cámaras", "Visualiza 4 cámaras"),
        ]
        for titulo, desc in pasos:
            row_p = tk.Frame(help_box, bg="#0F172A")
            row_p.pack(fill="x", pady=1)
            tk.Label(row_p, text=titulo, font=("Segoe UI", 8, "bold"), bg="#0F172A", fg=self.C_TEXT).pack(anchor="w")
            tk.Label(row_p, text=f"   {desc}", font=("Segoe UI", 8), bg="#0F172A", fg=self.C_MUTED).pack(anchor="w")

        # Estado ESP32 / 4 Relés abajo
        plc_card = tk.Frame(
            self.sidebar, bg="#0B1320",
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=14, pady=10
        )
        plc_card.pack(side="bottom", fill="x", padx=12, pady=14)

        tk.Label(plc_card, text="CONTROLADOR ESP32 · 4 RELÉS", font=("Segoe UI", 8, "bold"), bg="#0B1320", fg="#60A5FA").pack(anchor="w")
        plc_row = tk.Frame(plc_card, bg="#0B1320")
        plc_row.pack(anchor="w", pady=(3, 1))
        self.lbl_plc_dot = tk.Label(plc_row, text="●", font=("Segoe UI", 11), bg="#0B1320", fg=self.C_AMBER)
        self.lbl_plc_dot.pack(side="left", padx=(0, 6))
        self.lbl_plc_title = tk.Label(plc_row, text="ESP32 (USB + Wi-Fi)", font=("Segoe UI", 9, "bold"), bg="#0B1320", fg=self.C_TEXT)
        self.lbl_plc_title.pack(side="left")
        self.lbl_plc_ip = tk.Label(
            plc_card,
            text="R1:Alarma | R2:Corte | R3:OK | R4:Aux",
            font=("Segoe UI", 8), bg="#0B1320", fg=self.C_MUTED
        )
        self.lbl_plc_ip.pack(anchor="w")

        # ==================== 2. CONTENEDOR PRINCIPAL ====================
        self.main_area = tk.Frame(self.root, bg=self.C_BG_MAIN)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        # --- HEADER SUPERIOR ---
        self.header = tk.Frame(
            self.main_area, bg=self.C_BG_SIDEBAR, height=68,
            highlightbackground=self.C_BORDER, highlightthickness=1
        )
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_propagate(False)

        # PRIMERO empaquetamos a la derecha Paro de Emergencia y Sistema Armado para que SIEMPRE sean visibles y accesibles
        right_controls = tk.Frame(self.header, bg=self.C_BG_SIDEBAR)
        right_controls.pack(side="right", padx=14, pady=12)

        self.btn_armado = tk.Button(
            right_controls, text="●  SISTEMA ARMADO", font=("Segoe UI", 9, "bold"),
            bg="#064E3B", fg="#34D399", activebackground="#065F46", activeforeground="#34D399",
            bd=0, relief="flat", padx=11, pady=8, cursor="hand2",
            command=self._toggle_sistema_armado
        )
        self.btn_armado.pack(side="left", padx=(0, 8))

        self.btn_paro = tk.Button(
            right_controls, text="🛑 PARO DE EMERGENCIA", font=("Segoe UI", 9, "bold"),
            bg=self.C_RED, fg="white", activebackground="#DC2626", activeforeground="white",
            bd=0, relief="flat", padx=12, pady=8, cursor="hand2",
            command=self._toggle_paro_emergencia
        )
        self.btn_paro.pack(side="left")

        # Título de la celda a la izquierda
        self.lbl_cell_title = tk.Label(
            self.header, text=self.config["cell_name"],
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_SIDEBAR, fg=self.C_TEXT
        )
        self.lbl_cell_title.pack(side="left", padx=(14, 8))

        # Barra rápida de Cámaras Guardadas + IP en el Header
        quick_ip_frame = tk.Frame(
            self.header, bg="#0B1320",
            highlightbackground="#374151", highlightthickness=1,
            padx=8, pady=4
        )
        quick_ip_frame.pack(side="left", padx=4, pady=12)

        tk.Label(quick_ip_frame, text="📹 Cámara:", font=("Segoe UI", 8, "bold"), bg="#0B1320", fg="#60A5FA").pack(side="left", padx=(2, 4))

        # Selector desplegable de cámaras guardadas
        self.combo_saved_header = ttk.Combobox(quick_ip_frame, state="readonly", width=18, font=("Segoe UI", 9))
        self.combo_saved_header.pack(side="left", padx=(0, 5))
        self.combo_saved_header.bind("<<ComboboxSelected>>", self._on_header_saved_cam_selected)

        self.ent_quick_ip = tk.Entry(
            quick_ip_frame, font=("Consolas", 9), width=20,
            bg="#1F2937", fg="white", insertbackground="white", relief="flat"
        )
        self.ent_quick_ip.insert(0, self.config["camera_url"])
        self.ent_quick_ip.pack(side="left", padx=3, ipady=3)
        self.ent_quick_ip.bind("<Return>", lambda e: self._conectar_desde_barra_rapida())

        tk.Button(
            quick_ip_frame, text="Conectar", font=("Segoe UI", 8, "bold"),
            bg=self.C_BLUE, fg="white", activebackground="#1D4ED8", activeforeground="white",
            bd=0, relief="flat", padx=9, pady=3, cursor="hand2",
            command=self._conectar_desde_barra_rapida
        ).pack(side="left", padx=2)

        tk.Button(
            quick_ip_frame, text="🔍 Escanear", font=("Segoe UI", 8, "bold"),
            bg="#059669", fg="white", activebackground="#047857", activeforeground="white",
            bd=0, relief="flat", padx=9, pady=3, cursor="hand2",
            command=self._ir_y_escanear_camaras
        ).pack(side="left", padx=(3, 2))

        # --- CONTENEDOR DE VISTAS ---
        self.views_container = tk.Frame(self.main_area, bg=self.C_BG_MAIN)
        self.views_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=12)
        self.views_container.grid_rowconfigure(0, weight=1)
        self.views_container.grid_columnconfigure(0, weight=1)

        self.view_monitoreo = tk.Frame(self.views_container, bg=self.C_BG_MAIN)
        self.view_multicam = tk.Frame(self.views_container, bg=self.C_BG_MAIN)
        self.view_camaras = tk.Frame(self.views_container, bg=self.C_BG_MAIN)
        self.view_bitacora = tk.Frame(self.views_container, bg=self.C_BG_MAIN)
        self.view_calibracion = tk.Frame(self.views_container, bg=self.C_BG_MAIN)

        self._construir_vista_monitoreo()
        self._construir_vista_multicam()
        self._construir_vista_gestor_camaras()
        self._construir_vista_bitacora()
        self._construir_vista_calibracion()

        self._actualizar_combos_camaras_guardadas()
        self._cambiar_vista("monitoreo")

    # -----------------------------------------------------------------
    # VISTA 1: MONITOREO EN VIVO + GESTOR INTERACTIVO MULTI-GEOCERCA
    # -----------------------------------------------------------------
    def _construir_vista_monitoreo(self):
        self.view_monitoreo.grid_columnconfigure(0, weight=1)
        self.view_monitoreo.grid_columnconfigure(1, weight=0, minsize=345)
        self.view_monitoreo.grid_rowconfigure(0, weight=1)

        video_outer = tk.Frame(
            self.view_monitoreo, bg="#05080F",
            highlightbackground=self.C_BORDER, highlightthickness=1
        )
        video_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        video_outer.grid_rowconfigure(2, weight=1)
        video_outer.grid_columnconfigure(0, weight=1)

        # --- FILA 0: Selector de Modo + Estado de Cámara ---
        top_bar = tk.Frame(video_outer, bg="#0F172A", padx=10, pady=6)
        top_bar.grid(row=0, column=0, sticky="ew")

        tk.Label(top_bar, text="MODO:", font=("Segoe UI", 8, "bold"), bg="#0F172A", fg=self.C_MUTED).pack(side="left", padx=(0, 6))

        self.btn_mode_poly = tk.Button(
            top_bar, text="⬡ Geocercas", font=("Segoe UI", 9, "bold"),
            bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._cambiar_modo_zona("geocerca_poly")
        )
        self.btn_mode_poly.pack(side="left", padx=2)

        self.btn_mode_vline = tk.Button(
            top_bar, text="┃ Línea Vertical", font=("Segoe UI", 9, "bold"),
            bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._cambiar_modo_zona("linea_vertical")
        )
        self.btn_mode_vline.pack(side="left", padx=2)

        self.btn_mode_hline = tk.Button(
            top_bar, text="━ Línea Horizontal", font=("Segoe UI", 9, "bold"),
            bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            command=lambda: self._cambiar_modo_zona("linea_horizontal")
        )
        self.btn_mode_hline.pack(side="left", padx=2)

        self.btn_invertir = tk.Button(
            top_bar, text="⇄ Invertir", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#93C5FD", activebackground="#334155", activeforeground="white",
            bd=0, relief="flat", padx=9, pady=4, cursor="hand2",
            command=self._invertir_lado_peligro
        )
        self.btn_invertir.pack(side="left", padx=(8, 2))

        self.lbl_cam_status = tk.Label(
            top_bar, text="Conectando...", font=("Consolas", 9),
            bg="#0F172A", fg=self.C_AMBER
        )
        self.lbl_cam_status.pack(side="right", padx=6)

        # --- FILA 1: Barra de Zonas Limpia y Compacta ---
        self.poly_toolbar = tk.Frame(
            video_outer, bg="#111827",
            highlightbackground="#1F2937", highlightthickness=1,
            padx=8, pady=6
        )
        self.poly_toolbar.grid(row=1, column=0, sticky="ew")

        row_z_top = tk.Frame(self.poly_toolbar, bg="#111827")
        row_z_top.pack(fill="x")

        tk.Label(
            row_z_top, text="ZONA:",
            font=("Segoe UI", 8, "bold"), bg="#111827", fg="#60A5FA"
        ).pack(side="left", padx=(0, 5))

        self.combo_zonas = ttk.Combobox(
            row_z_top, state="readonly", width=18, font=("Segoe UI", 9, "bold")
        )
        self.combo_zonas.pack(side="left", padx=(0, 6))
        self.combo_zonas.bind("<<ComboboxSelected>>", self._on_combo_zona_selected)
        self.combo_zonas.bind("<ButtonPress-3>", lambda e: self._mostrar_menu_contextual_zona(e, self.selected_zone_idx))

        tk.Button(
            row_z_top, text="➕ Nueva", font=("Segoe UI", 8, "bold"),
            bg="#065F46", fg="#6EE7B7", activebackground="#047857", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=self._agregar_nueva_geocerca
        ).pack(side="left", padx=2)

        self.btn_add_pts = tk.Button(
            row_z_top, text="✏️ +Puntos", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#E5E7EB", activebackground="#2563EB", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=self._toggle_modo_agregar_puntos
        )
        self.btn_add_pts.pack(side="left", padx=2)

        self.btn_zone_safe_toggle = tk.Button(
            row_z_top, text="🔔 1: Alarma", font=("Segoe UI", 8, "bold"),
            bg="#78350F", fg="#FCD34D", activebackground="#92400E", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=lambda: self._alternar_inversion_zona(self.selected_zone_idx)
        )
        self.btn_zone_safe_toggle.pack(side="left", padx=2)

        tk.Button(
            row_z_top, text="🔔 Alarma (R1)", font=("Segoe UI", 8, "bold"),
            bg="#78350F", fg="#FDE68A", activebackground="#B45309", activeforeground="white",
            bd=0, relief="flat", padx=6, pady=4, cursor="hand2",
            command=lambda: self._set_accion_zona(self.selected_zone_idx, "alarma")
        ).pack(side="left", padx=1)

        tk.Button(
            row_z_top, text="⚡ Corte (R2)", font=("Segoe UI", 8, "bold"),
            bg="#7F1D1D", fg="#FECACA", activebackground="#B91C1C", activeforeground="white",
            bd=0, relief="flat", padx=6, pady=4, cursor="hand2",
            command=lambda: self._set_accion_zona(self.selected_zone_idx, "corte")
        ).pack(side="left", padx=1)

        tk.Button(
            row_z_top, text="📋 Copiar", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#CBD5E1", activebackground="#334155", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=lambda: self._copiar_geocerca(self.selected_zone_idx)
        ).pack(side="left", padx=2)

        tk.Button(
            row_z_top, text="📌 Pegar", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#CBD5E1", activebackground="#334155", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=lambda: self._pegar_geocerca()
        ).pack(side="left", padx=2)

        tk.Button(
            row_z_top, text="▢ 4 Puntos", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="#93C5FD", activebackground="#334155", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=self._resetear_zona_a_4_puntos
        ).pack(side="left", padx=2)

        tk.Button(
            row_z_top, text="🗑 Borrar", font=("Segoe UI", 8, "bold"),
            bg="#450A0A", fg="#FCA5A5", activebackground="#7F1D1D", activeforeground="white",
            bd=0, relief="flat", padx=7, pady=4, cursor="hand2",
            command=self._eliminar_geocerca_actual
        ).pack(side="left", padx=2)

        self.lbl_poly_hint = tk.Label(
            row_z_top,
            text="",
            font=("Segoe UI", 8), bg="#111827", fg="#9CA3AF"
        )
        self.lbl_poly_hint.pack(side="right", padx=4)

        # --- FILA 2: Canvas de Video Interactivo ---
        self.video_canvas = tk.Canvas(
            video_outer, bg="#05080F", highlightthickness=0, cursor="crosshair"
        )
        self.video_canvas.grid(row=2, column=0, sticky="nsew")

        self.video_canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self.video_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.video_canvas.bind("<ButtonPress-3>", self._on_canvas_right_click)
        self.video_canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.video_canvas.bind("<Motion>", self._on_canvas_hover)
        self.video_canvas.bind("<Configure>", self._on_canvas_resize)

        self._refrescar_botones_modo()
        self._refrescar_pills_zonas()

        # ==================== COLUMNA DERECHA (CARDS FIGMA) ====================
        right_col = tk.Frame(self.view_monitoreo, bg=self.C_BG_MAIN, width=345)
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.grid_propagate(False)
        right_col.grid_columnconfigure(0, weight=1)
        right_col.grid_rowconfigure(1, weight=1)

        self.interlock_card = tk.Frame(
            right_col, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=18, pady=16
        )
        self.interlock_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))

        tk.Label(
            self.interlock_card, text="ESTADO DEL ENCLAVAMIENTO",
            font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_MUTED
        ).pack(anchor="w", pady=(0, 12))

        self.status_box = tk.Frame(
            self.interlock_card, bg="#062E22",
            highlightbackground="#059669", highlightthickness=1,
            padx=14, pady=16
        )
        self.status_box.pack(fill="x", pady=(0, 14))

        self.lbl_status_icon = tk.Label(
            self.status_box, text="✓", font=("Segoe UI", 22, "bold"),
            bg="#059669", fg="white", width=2
        )
        self.lbl_status_icon.pack(pady=(0, 8))

        self.lbl_status_main = tk.Label(
            self.status_box, text="MÁQUINA OPERANDO",
            font=("Segoe UI", 13, "bold"), bg="#062E22", fg="#34D399"
        )
        self.lbl_status_main.pack()

        self.lbl_status_sub = tk.Label(
            self.status_box, text="Zonas Despejadas - Relé 3 (OK) Activo",
            font=("Segoe UI", 9), bg="#062E22", fg="#A7F3D0"
        )
        self.lbl_status_sub.pack(pady=(3, 0))

        # --- PANEL EN VIVO DE LOS 4 RELÉS ESP32 (CLIC PARA PROBAR) ---
        relay_box = tk.Frame(
            self.interlock_card, bg="#0B1320",
            highlightbackground="#1F2937", highlightthickness=1,
            padx=8, pady=8
        )
        relay_box.pack(fill="x", pady=(0, 10))

        rel_hdr = tk.Frame(relay_box, bg="#0B1320")
        rel_hdr.pack(fill="x", pady=(0, 5))
        tk.Label(
            rel_hdr, text="🔌 PLACA 4 RELÉS ESP32",
            font=("Segoe UI", 8, "bold"), bg="#0B1320", fg="#60A5FA"
        ).pack(side="left")
        self.lbl_esp32_conn_mini = tk.Label(
            rel_hdr, text="USB / Wi-Fi",
            font=("Consolas", 8), bg="#0B1320", fg=self.C_AMBER
        )
        self.lbl_esp32_conn_mini.pack(side="right")

        rel_grid = tk.Frame(relay_box, bg="#0B1320")
        rel_grid.pack(fill="x")
        for c_i in range(4):
            rel_grid.grid_columnconfigure(c_i, weight=1)

        self.relay_pills = []
        definicion_reles = [
            (1, "R1·ALARMA", "GPIO 26"),
            (2, "R2·CORTE",  "GPIO 27"),
            (3, "R3·VERDE",  "GPIO 14"),
            (4, "R4·AUX",    "GPIO 25"),
        ]
        for idx_r, (num_r, nom_r, pin_r) in enumerate(definicion_reles):
            btn_r = tk.Button(
                rel_grid,
                text=f"{nom_r}\nOFF",
                font=("Consolas", 8, "bold"),
                bg="#1E293B", fg="#9CA3AF",
                activebackground="#334155", activeforeground="white",
                bd=0, relief="flat", pady=4, cursor="hand2",
                command=lambda n=num_r: self._probar_rele_desde_ui(n)
            )
            btn_r.grid(row=0, column=idx_r, sticky="ew", padx=2)
            self.relay_pills.append((num_r, nom_r, btn_r))

        m1 = tk.Frame(self.interlock_card, bg=self.C_BG_CARD)
        m1.pack(fill="x", pady=3)
        tk.Label(m1, text="Velocidad Video / IA:", font=("Segoe UI", 9), bg=self.C_BG_CARD, fg=self.C_MUTED).pack(side="left")
        self.lbl_metric_fps = tk.Label(m1, text="0 FPS (IA: 0 ms)", font=("Consolas", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT)
        self.lbl_metric_fps.pack(side="right")

        m2 = tk.Frame(self.interlock_card, bg=self.C_BG_CARD)
        m2.pack(fill="x", pady=3)
        tk.Label(m2, text="Personas Detectadas:", font=("Segoe UI", 9), bg=self.C_BG_CARD, fg=self.C_MUTED).pack(side="left")
        self.lbl_metric_persons = tk.Label(m2, text="0 (0 en peligro)", font=("Consolas", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_GREEN)
        self.lbl_metric_persons.pack(side="right")

        m3 = tk.Frame(self.interlock_card, bg=self.C_BG_CARD)
        m3.pack(fill="x", pady=3)
        tk.Label(m3, text="Geocercas Activas:", font=("Segoe UI", 9), bg=self.C_BG_CARD, fg=self.C_MUTED).pack(side="left")
        self.lbl_metric_zones = tk.Label(m3, text="1 zona", font=("Consolas", 9, "bold"), bg=self.C_BG_CARD, fg="#60A5FA")
        self.lbl_metric_zones.pack(side="right")

        self.alerts_card = tk.Frame(
            right_col, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=18, pady=16
        )
        self.alerts_card.grid(row=1, column=0, sticky="nsew")
        self.alerts_card.grid_rowconfigure(1, weight=1)
        self.alerts_card.grid_columnconfigure(0, weight=1)

        hdr_al = tk.Frame(self.alerts_card, bg=self.C_BG_CARD)
        hdr_al.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        tk.Label(
            hdr_al, text="ALERTAS RECIENTES",
            font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_MUTED
        ).pack(side="left")
        tk.Button(
            hdr_al, text="Limpiar", font=("Segoe UI", 8),
            bg=self.C_BG_CARD2, fg=self.C_MUTED, bd=0, relief="flat", padx=8, pady=2,
            cursor="hand2", command=self._limpiar_alertas_recientes
        ).pack(side="right")

        self.alerts_list_frame = tk.Frame(self.alerts_card, bg=self.C_BG_CARD)
        self.alerts_list_frame.grid(row=1, column=0, sticky="nsew")

        self._renderizar_alertas_recientes()

    # -----------------------------------------------------------------
    # VISTA 2: MONITOR DE 4 CÁMARAS SIMULTÁNEAS (DASHBOARD 2x2 LIGERO)
    # -----------------------------------------------------------------
    def _construir_vista_multicam(self):
        self.view_multicam.grid_rowconfigure(1, weight=1)
        self.view_multicam.grid_columnconfigure(0, weight=1)

        # Barra superior del Monitor 4x
        top_mc = tk.Frame(
            self.view_multicam, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=18, pady=10
        )
        top_mc.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            top_mc, text="📺 Monitor Multi-Cámara",
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT
        ).pack(side="left")

        tk.Button(
            top_mc, text="📡 Gestor de Cámaras", font=("Segoe UI", 9, "bold"),
            bg=self.C_BLUE, fg="white", bd=0, relief="flat", padx=12, pady=5, cursor="hand2",
            command=lambda: self._cambiar_vista("camaras")
        ).pack(side="right", padx=(6, 0))

        tk.Button(
            top_mc, text="⚡ Auto-Asignar", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="white", bd=0, relief="flat", padx=12, pady=5, cursor="hand2",
            command=self._auto_asignar_4_camaras
        ).pack(side="right", padx=6)

        self.btn_quad_ai = tk.Button(
            top_mc, text="👁️ Solo Visualización", font=("Segoe UI", 9, "bold"),
            bg="#1E293B", fg="#38BDF8", bd=0, relief="flat", padx=12, pady=5, cursor="hand2",
            command=self._toggle_quad_ai
        )
        self.btn_quad_ai.pack(side="right", padx=6)

        # Cuadrícula 2x2
        grid_frame = tk.Frame(self.view_multicam, bg=self.C_BG_MAIN)
        grid_frame.grid(row=1, column=0, sticky="nsew")
        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_rowconfigure(1, weight=1)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        self.quad_slots_ui = []
        for idx in range(4):
            r, c = divmod(idx, 2)
            cell = tk.Frame(
                grid_frame, bg="#080C14",
                highlightbackground=self.C_BORDER, highlightthickness=1
            )
            cell.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            cell.grid_rowconfigure(1, weight=1)
            cell.grid_columnconfigure(0, weight=1)

            bar = tk.Frame(cell, bg="#111827", padx=8, pady=5)
            bar.grid(row=0, column=0, sticky="ew")

            tk.Label(
                bar, text=f"CAM {idx + 1}:", font=("Segoe UI", 8, "bold"),
                bg="#111827", fg="#60A5FA"
            ).pack(side="left", padx=(0, 4))

            combo = ttk.Combobox(bar, state="readonly", width=26, font=("Segoe UI", 8))
            combo.pack(side="left", padx=2)
            combo.bind("<<ComboboxSelected>>", lambda e, i=idx: self._on_quad_cam_selected(i))

            lbl_st = tk.Label(bar, text="○ Apagada", font=("Consolas", 8), bg="#111827", fg=self.C_MUTED)
            lbl_st.pack(side="left", padx=8)

            tk.Button(
                bar, text="✖", font=("Segoe UI", 8, "bold"),
                bg="#450A0A", fg="#FCA5A5", bd=0, relief="flat", padx=6, pady=1, cursor="hand2",
                command=lambda i=idx: self._apagar_slot_quad(i)
            ).pack(side="right", padx=2)

            tk.Button(
                bar, text="🔄", font=("Segoe UI", 8),
                bg="#1E293B", fg="white", bd=0, relief="flat", padx=6, pady=1, cursor="hand2",
                command=lambda i=idx: self._reconectar_slot_quad(i)
            ).pack(side="right", padx=2)

            tk.Button(
                bar, text="🎯 Usar en IA", font=("Segoe UI", 8, "bold"),
                bg="#1E3A8A", fg="#93C5FD", bd=0, relief="flat", padx=7, pady=1, cursor="hand2",
                command=lambda i=idx: self._enviar_slot_a_principal(i)
            ).pack(side="right", padx=2)

            cv_quad = tk.Canvas(cell, bg="#05080F", highlightthickness=0)
            cv_quad.grid(row=1, column=0, sticky="nsew")
            cv_quad.bind("<Double-Button-1>", lambda e, i=idx: self._enviar_slot_a_principal(i))

            self.quad_slots_ui.append({
                "combo": combo,
                "lbl_status": lbl_st,
                "canvas": cv_quad,
                "last_frame_id": -1
            })

    # -----------------------------------------------------------------
    # VISTA 3: GESTOR DE CÁMARAS Y ESCÁNER DE RED (USB, WI-FI, ETHERNET, AUTH)
    # -----------------------------------------------------------------
    def _construir_vista_gestor_camaras(self):
        self.view_camaras.grid_columnconfigure(0, weight=1)
        self.view_camaras.grid_columnconfigure(1, weight=1)
        self.view_camaras.grid_rowconfigure(0, weight=1)

        # ==================== PANEL IZQUIERDO: ESCÁNER DE RED Y USB ====================
        left_scan = tk.Frame(
            self.view_camaras, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=18, pady=16
        )
        left_scan.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_scan.grid_rowconfigure(3, weight=1)
        left_scan.grid_columnconfigure(0, weight=1)

        hdr_s = tk.Frame(left_scan, bg=self.C_BG_CARD)
        hdr_s.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(
            hdr_s, text="📡 Escáner de Red y Cámaras",
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_CARD, fg="#38BDF8"
        ).pack(side="left")

        self.btn_scan_net = tk.Button(
            hdr_s, text="🔍 Escanear Ahora", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="white", activebackground="#047857", activeforeground="white",
            bd=0, relief="flat", padx=14, pady=6, cursor="hand2",
            command=self._iniciar_escaneo_completo
        )
        self.btn_scan_net.pack(side="right")

        # Fila de filtro de subred opcional + Auto-conectar
        opt_s = tk.Frame(left_scan, bg="#0F172A", padx=10, pady=8)
        opt_s.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        tk.Label(opt_s, text="Subred / IP extra:", font=("Segoe UI", 8, "bold"), bg="#0F172A", fg=self.C_MUTED).pack(side="left")
        self.ent_subred_extra = tk.Entry(opt_s, font=("Consolas", 9), width=16, bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_subred_extra.pack(side="left", padx=6, ipady=2)

        self.var_autoconnect_scan = tk.BooleanVar(value=bool(self.config.get("auto_connect_open_cams", True)))
        tk.Checkbutton(
            opt_s, text="⚡ Auto-conectar cámaras abiertas",
            variable=self.var_autoconnect_scan,
            bg="#0F172A", fg="#6EE7B7", selectcolor="#1E293B",
            activebackground="#0F172A", font=("Segoe UI", 8, "bold"),
            command=self._on_toggle_autoconnect
        ).pack(side="right")

        self.lbl_scan_progress = tk.Label(
            left_scan,
            text="Pulsa '🔍 Escanear Ahora' para buscar cámaras en la red.",
            font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_MUTED, anchor="w"
        )
        self.lbl_scan_progress.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        # Lista scrollable de cámaras encontradas en el escaneo
        scan_box_outer = tk.Frame(left_scan, bg="#0B1320", highlightbackground="#1F2937", highlightthickness=1)
        scan_box_outer.grid(row=3, column=0, sticky="nsew")
        scan_box_outer.grid_rowconfigure(0, weight=1)
        scan_box_outer.grid_columnconfigure(0, weight=1)

        self.scan_canvas = tk.Canvas(scan_box_outer, bg="#0B1320", highlightthickness=0)
        scan_scroll = ttk.Scrollbar(scan_box_outer, orient="vertical", command=self.scan_canvas.yview)
        self.scan_list_inner = tk.Frame(self.scan_canvas, bg="#0B1320", padx=8, pady=8)

        self.scan_list_inner.bind(
            "<Configure>",
            lambda e: self.scan_canvas.configure(scrollregion=self.scan_canvas.bbox("all"))
        )
        self.scan_canvas_win = self.scan_canvas.create_window((0, 0), window=self.scan_list_inner, anchor="nw")
        self.scan_canvas.bind(
            "<Configure>",
            lambda e: self.scan_canvas.itemconfig(self.scan_canvas_win, width=e.width)
        )
        self.scan_canvas.configure(yscrollcommand=scan_scroll.set)
        self.scan_canvas.grid(row=0, column=0, sticky="nsew")
        scan_scroll.grid(row=0, column=1, sticky="ns")

        # ==================== PANEL DERECHO: REGISTRO CRUD + CÁMARAS GUARDADAS ====================
        right_crud = tk.Frame(
            self.view_camaras, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=18, pady=16
        )
        right_crud.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_crud.grid_rowconfigure(2, weight=1)
        right_crud.grid_columnconfigure(0, weight=1)

        # --- Formulario de Registro / Modificación ---
        form_hdr = tk.Frame(right_crud, bg=self.C_BG_CARD)
        form_hdr.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.lbl_form_title = tk.Label(
            form_hdr, text="➕ Registrar / Modificar Cámara",
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_CARD, fg="#34D399"
        )
        self.lbl_form_title.pack(side="left")

        tk.Button(
            form_hdr, text="🆕 Limpiar", font=("Segoe UI", 8, "bold"),
            bg="#1E293B", fg="white", bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._limpiar_formulario_camara
        ).pack(side="right")

        form_box = tk.Frame(right_crud, bg="#0F172A", highlightbackground="#1F2937", highlightthickness=1, padx=14, pady=12)
        form_box.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        form_box.grid_columnconfigure(1, weight=1)
        form_box.grid_columnconfigure(3, weight=1)

        # Fila 0: Nombre y Perfil
        tk.Label(form_box, text="Nombre:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg=self.C_TEXT).grid(row=0, column=0, sticky="w", pady=4)
        self.ent_cam_name = tk.Entry(form_box, font=("Segoe UI", 9), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_name.grid(row=0, column=1, sticky="ew", padx=(6, 12), pady=4, ipady=4)

        tk.Label(form_box, text="Tipo / Marca:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg=self.C_TEXT).grid(row=0, column=2, sticky="w", pady=4)
        self.combo_cam_profile = ttk.Combobox(
            form_box, state="readonly", font=("Segoe UI", 9),
            values=[p[0] for p in CAMERA_PROFILES]
        )
        self.combo_cam_profile.current(0)
        self.combo_cam_profile.grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=4)
        self.combo_cam_profile.bind("<<ComboboxSelected>>", self._on_cam_profile_changed)

        # Fila 1: IP/Host/USB y Puerto
        tk.Label(form_box, text="IP o USB:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg=self.C_TEXT).grid(row=1, column=0, sticky="w", pady=4)
        self.ent_cam_host = tk.Entry(form_box, font=("Consolas", 10), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_host.grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=4, ipady=4)

        tk.Label(form_box, text="Puerto:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg=self.C_TEXT).grid(row=1, column=2, sticky="w", pady=4)
        self.ent_cam_port = tk.Entry(form_box, font=("Consolas", 10), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_port.insert(0, "8080")
        self.ent_cam_port.grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=4, ipady=4)

        # Fila 2: Usuario y Contraseña
        tk.Label(form_box, text="Usuario:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg="#FBBF24").grid(row=2, column=0, sticky="w", pady=4)
        self.ent_cam_user = tk.Entry(form_box, font=("Consolas", 9), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_user.grid(row=2, column=1, sticky="ew", padx=(6, 12), pady=4, ipady=4)

        tk.Label(form_box, text="Contraseña:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg="#FBBF24").grid(row=2, column=2, sticky="w", pady=4)
        pwd_frame = tk.Frame(form_box, bg="#0F172A")
        pwd_frame.grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=4)
        pwd_frame.grid_columnconfigure(0, weight=1)
        self.ent_cam_pwd = tk.Entry(pwd_frame, font=("Consolas", 9), show="*", bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_pwd.grid(row=0, column=0, sticky="ew", ipady=4)
        self.mostrar_pwd = False
        tk.Button(
            pwd_frame, text="👁", font=("Segoe UI", 8), bg="#334155", fg="white", bd=0, relief="flat", padx=5,
            command=self._toggle_ver_password
        ).grid(row=0, column=1, padx=(3, 0))

        # Fila 3: Ruta de Stream
        tk.Label(form_box, text="Ruta Video:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg=self.C_TEXT).grid(row=3, column=0, sticky="w", pady=4)
        self.ent_cam_path = tk.Entry(form_box, font=("Consolas", 9), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_path.insert(0, "/video")
        self.ent_cam_path.grid(row=3, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=4, ipady=4)

        # Fila 4: Botones de Acción CRUD
        btn_crud_row = tk.Frame(form_box, bg="#0F172A")
        btn_crud_row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 2))

        tk.Button(
            btn_crud_row, text="💾 Guardar Cámara", font=("Segoe UI", 9, "bold"),
            bg=self.C_GREEN, fg="white", activebackground="#059669", activeforeground="white",
            bd=0, relief="flat", padx=14, pady=6, cursor="hand2",
            command=lambda: self._guardar_camara_formulario(conectar_ahora=False)
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            btn_crud_row, text="⚡ Guardar y Conectar", font=("Segoe UI", 9, "bold"),
            bg=self.C_BLUE, fg="white", activebackground="#1D4ED8", activeforeground="white",
            bd=0, relief="flat", padx=14, pady=6, cursor="hand2",
            command=lambda: self._guardar_camara_formulario(conectar_ahora=True)
        ).pack(side="left")

        # --- Lista de Cámaras Guardadas ---
        saved_outer = tk.Frame(right_crud, bg="#0B1320", highlightbackground="#1F2937", highlightthickness=1)
        saved_outer.grid(row=2, column=0, sticky="nsew")
        saved_outer.grid_rowconfigure(1, weight=1)
        saved_outer.grid_columnconfigure(0, weight=1)

        saved_hdr = tk.Frame(saved_outer, bg="#111827", padx=12, pady=8)
        saved_hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(
            saved_hdr, text="📋 Cámaras Guardadas",
            font=("Segoe UI", 10, "bold"), bg="#111827", fg="#60A5FA"
        ).pack(side="left")

        self.saved_canvas = tk.Canvas(saved_outer, bg="#0B1320", highlightthickness=0)
        saved_scroll = ttk.Scrollbar(saved_outer, orient="vertical", command=self.saved_canvas.yview)
        self.saved_list_inner = tk.Frame(self.saved_canvas, bg="#0B1320", padx=8, pady=8)

        self.saved_list_inner.bind(
            "<Configure>",
            lambda e: self.saved_canvas.configure(scrollregion=self.saved_canvas.bbox("all"))
        )
        self.saved_canvas_win = self.saved_canvas.create_window((0, 0), window=self.saved_list_inner, anchor="nw")
        self.saved_canvas.bind(
            "<Configure>",
            lambda e: self.saved_canvas.itemconfig(self.saved_canvas_win, width=e.width)
        )
        self.saved_canvas.configure(yscrollcommand=saved_scroll.set)
        self.saved_canvas.grid(row=1, column=0, sticky="nsew")
        saved_scroll.grid(row=1, column=1, sticky="ns")

        self._renderizar_lista_camaras_guardadas()

    # -----------------------------------------------------------------
    # VISTA 4: BITÁCORA DE EVENTOS
    # -----------------------------------------------------------------
    def _construir_vista_bitacora(self):
        self.view_bitacora.grid_rowconfigure(1, weight=1)
        self.view_bitacora.grid_columnconfigure(0, weight=1)

        top = tk.Frame(
            self.view_bitacora, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=20, pady=14
        )
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        tk.Label(
            top, text="📋 Bitácora de Eventos e Intrusiones",
            font=("Segoe UI", 13, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT
        ).pack(side="left")

        tk.Button(
            top, text="📥 Exportar CSV", font=("Segoe UI", 9, "bold"),
            bg=self.C_BLUE, fg="white", bd=0, relief="flat", padx=14, pady=7,
            cursor="hand2", command=self._exportar_bitacora_csv
        ).pack(side="right", padx=(8, 0))

        tk.Button(
            top, text="🗑️ Limpiar Historial", font=("Segoe UI", 9, "bold"),
            bg="#374151", fg="white", bd=0, relief="flat", padx=14, pady=7,
            cursor="hand2", command=self._limpiar_bitacora
        ).pack(side="right")

        table_frame = tk.Frame(
            self.view_bitacora, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1
        )
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Safe.Treeview",
            background="#111827", foreground="#F9FAFB",
            fieldbackground="#111827", rowheight=30, font=("Segoe UI", 9)
        )
        style.configure(
            "Safe.Treeview.Heading",
            background="#1E293B", foreground="#9CA3AF",
            font=("Segoe UI", 9, "bold")
        )

        cols = ("fecha", "hora", "tipo", "titulo", "detalle")
        self.tree_log = ttk.Treeview(table_frame, columns=cols, show="headings", style="Safe.Treeview")
        self.tree_log.heading("fecha", text="Fecha")
        self.tree_log.heading("hora", text="Hora")
        self.tree_log.heading("tipo", text="Nivel")
        self.tree_log.heading("titulo", text="Evento")
        self.tree_log.heading("detalle", text="Detalle / Zona")

        self.tree_log.column("fecha", width=105, anchor="center")
        self.tree_log.column("hora", width=95, anchor="center")
        self.tree_log.column("tipo", width=95, anchor="center")
        self.tree_log.column("titulo", width=230, anchor="w")
        self.tree_log.column("detalle", width=480, anchor="w")
        self.tree_log.grid(row=0, column=0, sticky="nsew")

    # -----------------------------------------------------------------
    # VISTA 5: CONFIGURACIÓN SIMPLIFICADA Y VISUAL
    # -----------------------------------------------------------------
    def _construir_vista_calibracion(self):
        self.view_calibracion.grid_columnconfigure(0, weight=1)
        self.view_calibracion.grid_columnconfigure(1, weight=1)

        header_cfg = tk.Frame(
            self.view_calibracion, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=20, pady=14
        )
        header_cfg.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        tk.Label(
            header_cfg, text="⚙️ Configuración del Sistema",
            font=("Segoe UI", 13, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT
        ).pack(side="left")

        tk.Button(
            header_cfg, text="💾 Guardar Cambios", font=("Segoe UI", 10, "bold"),
            bg=self.C_GREEN, fg="white", activebackground="#059669", activeforeground="white",
            bd=0, relief="flat", padx=18, pady=7, cursor="hand2",
            command=self._guardar_calibracion_ui
        ).pack(side="right")

        card_cam = tk.Frame(
            self.view_calibracion, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=22, pady=18
        )
        card_cam.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 14))

        tk.Label(
            card_cam, text="1. Cámara Activa y Fluidez",
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_CARD, fg="#60A5FA"
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(card_cam, text="Dirección IP o URL de Cámara Principal:", font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT).pack(anchor="w")
        cam_input_row = tk.Frame(card_cam, bg=self.C_BG_CARD)
        cam_input_row.pack(fill="x", pady=(4, 10))
        self.ent_cam_url = tk.Entry(cam_input_row, font=("Consolas", 11), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cam_url.insert(0, self.config["camera_url"])
        self.ent_cam_url.pack(side="left", fill="x", expand=True, ipady=6)

        tk.Button(
            cam_input_row, text="📡 Gestor / Escáner", font=("Segoe UI", 9, "bold"),
            bg="#059669", fg="white", activebackground="#047857", activeforeground="white",
            bd=0, relief="flat", padx=12, pady=6, cursor="hand2",
            command=self._ir_y_escanear_camaras
        ).pack(side="left", padx=(8, 0))

        mini_steps = tk.Frame(card_cam, bg="#0F172A", padx=12, pady=10)
        mini_steps.pack(fill="x", pady=(0, 14))
        tk.Label(mini_steps, text="📱 Resumen de Conexión:", font=("Segoe UI", 9, "bold"), bg="#0F172A", fg="#38BDF8").pack(anchor="w", pady=(0, 4))
        tk.Label(mini_steps, text="• Usa '📡 Gestor de Cámaras' para buscar cámaras Wi-Fi, Ethernet o USB.", font=("Segoe UI", 9), bg="#0F172A", fg=self.C_TEXT).pack(anchor="w")
        tk.Label(mini_steps, text="• Guarda tus cámaras con nombre y clave para elegirlas con 1 clic.", font=("Segoe UI", 9), bg="#0F172A", fg=self.C_TEXT).pack(anchor="w")

        tk.Label(card_cam, text="Modo de Fluidez de Video:", font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT).pack(anchor="w", pady=(2, 4))
        self.var_max_width = tk.IntVar(value=int(self.config.get("max_width", 640)))
        res_frame = tk.Frame(card_cam, bg=self.C_BG_CARD)
        res_frame.pack(fill="x", pady=(0, 12))
        for txt, val in [("⚡ Ultra Rápido", 640), ("⚖️ Equilibrado", 800), ("🔍 Alta Definición", 960)]:
            tk.Radiobutton(
                res_frame, text=txt, variable=self.var_max_width, value=val,
                bg=self.C_BG_CARD, fg=self.C_TEXT, selectcolor="#1E293B",
                activebackground=self.C_BG_CARD, activeforeground="white",
                font=("Segoe UI", 9)
            ).pack(anchor="w", pady=1)

        tk.Label(card_cam, text="Nombre de la Celda / Estación:", font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT).pack(anchor="w")
        self.ent_cell_name = tk.Entry(card_cam, font=("Segoe UI", 10), bg="#1E293B", fg="white", insertbackground="white", relief="flat")
        self.ent_cell_name.insert(0, self.config["cell_name"])
        self.ent_cell_name.pack(fill="x", pady=(4, 12), ipady=5)

        # Apartado de Modo de Prueba con la Cámara de la PC
        test_pc_box = tk.Frame(
            card_cam, bg="#0F172A",
            highlightbackground="#334155", highlightthickness=1,
            padx=12, pady=10
        )
        test_pc_box.pack(fill="x", pady=(4, 2))
        tk.Label(
            test_pc_box, text="💻 Modo de Prueba:",
            font=("Segoe UI", 9, "bold"), bg="#0F172A", fg="#FBBF24"
        ).pack(anchor="w")
        tk.Label(
            test_pc_box,
            text="La cámara de la PC permanece apagada por defecto. Actívala únicamente para pruebas.",
            font=("Segoe UI", 8), bg="#0F172A", fg=self.C_MUTED
        ).pack(anchor="w", pady=(2, 6))

        self.btn_test_pc_cam = tk.Button(
            test_pc_box, text="🧪 Modo Prueba: OFF", font=("Segoe UI", 9, "bold"),
            bg="#374151", fg="#D1D5DB", activebackground="#4B5563", activeforeground="white",
            bd=0, relief="flat", padx=12, pady=6, cursor="hand2",
            command=self._toggle_modo_prueba_pc
        )
        self.btn_test_pc_cam.pack(anchor="w")

        card_det = tk.Frame(
            self.view_calibracion, bg=self.C_BG_CARD,
            highlightbackground=self.C_BORDER, highlightthickness=1,
            padx=22, pady=18
        )
        card_det.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 14))

        tk.Label(
            card_det, text="2. Reglas de Detección y ESP32 (4 Relés)",
            font=("Segoe UI", 12, "bold"), bg=self.C_BG_CARD, fg="#34D399"
        ).pack(anchor="w", pady=(0, 10))

        tk.Label(card_det, text="Punto del cuerpo para activar alarma/corte:", font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT).pack(anchor="w", pady=(0, 3))
        self.var_detect_pt = tk.StringVar(value=self.config.get("detect_point", "pies"))
        pt_frame = tk.Frame(card_det, bg=self.C_BG_CARD)
        pt_frame.pack(fill="x", pady=(0, 8))
        tk.Radiobutton(
            pt_frame, text="👟 Pies — Recomendado para piso",
            variable=self.var_detect_pt, value="pies",
            bg=self.C_BG_CARD, fg=self.C_TEXT, selectcolor="#1E293B",
            activebackground=self.C_BG_CARD, font=("Segoe UI", 9)
        ).pack(anchor="w")
        tk.Radiobutton(
            pt_frame, text="🎯 Centro del cuerpo — Recomendado para cámaras altas",
            variable=self.var_detect_pt, value="centro",
            bg=self.C_BG_CARD, fg=self.C_TEXT, selectcolor="#1E293B",
            activebackground=self.C_BG_CARD, font=("Segoe UI", 9)
        ).pack(anchor="w")

        tk.Label(card_det, text="Sensibilidad mínima IA:", font=("Segoe UI", 9, "bold"), bg=self.C_BG_CARD, fg=self.C_TEXT).pack(anchor="w")
        self.scale_conf = tk.Scale(
            card_det, from_=30, to=90, orient="horizontal",
            bg=self.C_BG_CARD, fg=self.C_TEXT, highlightthickness=0, troughcolor="#1E293B"
        )
        self.scale_conf.set(self.config["confidence"])
        self.scale_conf.pack(fill="x", pady=(2, 8))

        self.var_sound = tk.BooleanVar(value=self.config["sound_alarm"])
        tk.Checkbutton(
            card_det, text="🔊 Sonido de alerta en PC al detectar intrusión",
            variable=self.var_sound, bg=self.C_BG_CARD, fg=self.C_TEXT,
            selectcolor="#1E293B", activebackground=self.C_BG_CARD, font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=2)

        self.var_epp = tk.BooleanVar(value=self.config["require_epp"])

        # --- PANEL CONFIGURACIÓN ESP32 + PLACA DE 4 RELÉS (100% USB SERIAL) ---
        esp_box = tk.Frame(
            card_det, bg="#0F172A",
            highlightbackground="#334155", highlightthickness=1,
            padx=12, pady=10
        )
        esp_box.pack(fill="x", pady=(8, 4))

        tk.Label(
            esp_box, text="🔌 Conexión Directa ESP32 por Cable USB (Puerto Serial COM)",
            font=("Segoe UI", 9, "bold"), bg="#0F172A", fg="#38BDF8"
        ).pack(anchor="w", pady=(0, 6))

        self.var_esp32_mode = tk.StringVar(value="usb")
        self.ent_esp32_ip = tk.Entry(esp_box)

        row_com = tk.Frame(esp_box, bg="#0F172A")
        row_com.pack(fill="x", pady=4)
        tk.Label(row_com, text="Puerto USB COM:", font=("Segoe UI", 8, "bold"), bg="#0F172A", fg=self.C_TEXT, width=15, anchor="w").pack(side="left")
        self.combo_esp32_port = ttk.Combobox(row_com, width=28, font=("Consolas", 9))
        self.combo_esp32_port.pack(side="left", padx=(0, 6))
        self.combo_esp32_port.bind("<<ComboboxSelected>>", self._on_esp32_port_selected)
        tk.Button(
            row_com, text="🔍 Detectar y Conectar USB", font=("Segoe UI", 8, "bold"),
            bg="#059669", fg="white", bd=0, relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._detectar_y_reconectar_usb
        ).pack(side="left")

        self.var_esp32_low = tk.BooleanVar(value=bool(self.config.get("esp32_active_low", True)))
        tk.Checkbutton(
            esp_box, text="Módulo de 4 Relés Activo en LOW (Estándar optoacoplado)",
            variable=self.var_esp32_low, bg="#0F172A", fg=self.C_MUTED,
            selectcolor="#1E293B", activebackground="#0F172A", font=("Segoe UI", 8),
            command=self._on_esp32_port_selected
        ).pack(anchor="w", pady=(4, 6))

        tk.Label(
            esp_box,
            text="Pines ESP32: GPIO26=IN1(Alarma) | GPIO27=IN2(Corte) | GPIO14=IN3(Verde) | GPIO25=IN4(Aux)",
            font=("Consolas", 8), bg="#0F172A", fg="#93C5FD"
        ).pack(anchor="w", pady=(0, 6))

        test_rel_row = tk.Frame(esp_box, bg="#0F172A")
        test_rel_row.pack(fill="x")
        for n_r, txt_r, col_r in [
            (1, "🔔 Probar R1 (Alarma)", "#D97706"),
            (2, "⚡ Probar R2 (Corte)", "#DC2626"),
            (3, "🟢 Probar R3 (Verde)", "#059669"),
            (4, "🔧 Probar R4 (Aux)", "#2563EB"),
        ]:
            tk.Button(
                test_rel_row, text=txt_r, font=("Segoe UI", 8, "bold"),
                bg=col_r, fg="white", bd=0, relief="flat", padx=6, pady=4, cursor="hand2",
                command=lambda nr=n_r: self._probar_rele_desde_ui(nr)
            ).pack(side="left", expand=True, fill="x", padx=2)

        self.ent_plc_ip = tk.Entry(card_det)
        self.ent_plc_ip.insert(0, self.config["plc_ip"])
        self.ent_plc_port = tk.Entry(card_det)
        self.ent_plc_port.insert(0, self.config["plc_port"])

        self._refrescar_puertos_com_ui()

    # -----------------------------------------------------------------
    # LÓGICA DEL GESTOR DE CÁMARAS (CRUD + ESCÁNER + AUTO-CONEXIÓN)
    # -----------------------------------------------------------------
    def _ir_y_escanear_camaras(self):
        self._cambiar_vista("camaras")
        self._iniciar_escaneo_completo()

    def _on_toggle_autoconnect(self):
        self.config["auto_connect_open_cams"] = bool(self.var_autoconnect_scan.get())
        self._guardar_config()

    def _toggle_ver_password(self):
        self.mostrar_pwd = not self.mostrar_pwd
        self.ent_cam_pwd.configure(show="" if self.mostrar_pwd else "*")

    def _on_cam_profile_changed(self, event=None):
        idx = self.combo_cam_profile.current()
        if 0 <= idx < len(CAMERA_PROFILES):
            _, tipo, port, path = CAMERA_PROFILES[idx]
            self.ent_cam_port.delete(0, tk.END)
            self.ent_cam_port.insert(0, port)
            self.ent_cam_path.delete(0, tk.END)
            self.ent_cam_path.insert(0, path)

    def _limpiar_formulario_camara(self):
        self.editing_camera_id = None
        self.lbl_form_title.configure(text="➕ Registrar Nueva Cámara", fg="#34D399")
        self.ent_cam_name.delete(0, tk.END)
        self.ent_cam_host.delete(0, tk.END)
        self.ent_cam_port.delete(0, tk.END)
        self.ent_cam_port.insert(0, "8080")
        self.ent_cam_user.delete(0, tk.END)
        self.ent_cam_pwd.delete(0, tk.END)
        self.ent_cam_path.delete(0, tk.END)
        self.ent_cam_path.insert(0, "/video")
        self.combo_cam_profile.current(0)

    def _cargar_camara_en_formulario(self, cam_dict, es_edicion=True):
        host_cam = str(cam_dict.get("host", "")).strip()
        guardada_previa = next(
            (c for c in self.config.get("saved_cameras", []) if str(c.get("host", "")).strip() == host_cam and host_cam),
            None
        )
        if es_edicion and "id" in cam_dict:
            self.editing_camera_id = cam_dict["id"]
            self.lbl_form_title.configure(text=f"✏️ Modificando: {cam_dict.get('name', '')}", fg="#FBBF24")
        elif guardada_previa and "id" in guardada_previa:
            self.editing_camera_id = guardada_previa["id"]
            self.lbl_form_title.configure(text=f"🔑 Configurar Cámara ({host_cam})", fg="#38BDF8")
        else:
            self.editing_camera_id = None
            self.lbl_form_title.configure(text="🔑 Configurar y Guardar Cámara Detectada", fg="#38BDF8")

        self.ent_cam_name.delete(0, tk.END)
        self.ent_cam_name.insert(0, cam_dict.get("name") or (guardada_previa.get("name") if guardada_previa else "Cámara Nueva"))

        self.ent_cam_host.delete(0, tk.END)
        self.ent_cam_host.insert(0, host_cam)

        self.ent_cam_port.delete(0, tk.END)
        self.ent_cam_port.insert(0, cam_dict.get("port") or (guardada_previa.get("port") if guardada_previa else "554"))

        user_val = cam_dict.get("username") or (guardada_previa.get("username") if guardada_previa else "")
        if not user_val and cam_dict.get("type") == "rtsp":
            user_val = "admin"
        pwd_val = cam_dict.get("password") or (guardada_previa.get("password") if guardada_previa else "")

        self.ent_cam_user.delete(0, tk.END)
        self.ent_cam_user.insert(0, user_val)

        self.ent_cam_pwd.delete(0, tk.END)
        self.ent_cam_pwd.insert(0, pwd_val)

        self.ent_cam_path.delete(0, tk.END)
        self.ent_cam_path.insert(0, cam_dict.get("stream_path") or (guardada_previa.get("stream_path") if guardada_previa else "/Streaming/Channels/102"))

        if cam_dict.get("requires_auth"):
            self.ent_cam_pwd.focus_set() if user_val else self.ent_cam_user.focus_set()

    def _guardar_camara_formulario(self, conectar_ahora=False):
        nombre = self.ent_cam_name.get().strip()
        host = self.ent_cam_host.get().strip()
        port = self.ent_cam_port.get().strip()
        user = self.ent_cam_user.get().strip()
        pwd = self.ent_cam_pwd.get().strip()
        path = self.ent_cam_path.get().strip()

        if not host:
            messagebox.showwarning("Dato faltante", "Ingresa la dirección IP de la cámara o el número de puerto USB (0, 1...).")
            return

        if not nombre:
            nombre = f"Webcam USB #{host}" if host.isdigit() else f"Cámara ({host})"

        prof_idx = self.combo_cam_profile.current()
        tipo = CAMERA_PROFILES[prof_idx][1] if 0 <= prof_idx < len(CAMERA_PROFILES) else "http"
        if host.isdigit():
            tipo = "usb"
        elif port in ("554", "8554") or host.startswith("rtsp://"):
            tipo = "rtsp"

        datos = {
            "id": self.editing_camera_id or f"cam_{uuid.uuid4().hex[:8]}",
            "name": nombre,
            "type": tipo,
            "host": host,
            "port": port,
            "username": user,
            "password": pwd,
            "stream_path": path,
        }
        datos["url"] = construir_url_camara(datos)

        saved_list = self.config.setdefault("saved_cameras", [])
        actualizado = False
        for i, c in enumerate(saved_list):
            if c.get("id") == datos["id"] or (not self.editing_camera_id and c.get("url") == datos["url"]):
                datos["id"] = c.get("id", datos["id"])
                saved_list[i] = datos
                actualizado = True
                break

        if not actualizado:
            saved_list.append(datos)

        self._guardar_config()
        self._renderizar_lista_camaras_guardadas()
        self._actualizar_combos_camaras_guardadas()
        self._limpiar_formulario_camara()

        accion_str = "modificada" if actualizado else "almacenada"
        self._registrar_evento("INFO", "Cámara Guardada", f"'{nombre}' {accion_str} ({datos['url']})")

        if conectar_ahora:
            self._activar_camara_como_principal(datos["url"], nombre)

    def _eliminar_camara_guardada(self, cam_id):
        saved_list = self.config.get("saved_cameras", [])
        cam_obj = next((c for c in saved_list if c.get("id") == cam_id), None)
        if not cam_obj:
            return
        if not messagebox.askyesno("Eliminar Cámara", f"¿Seguro que deseas borrar '{cam_obj.get('name')}' de tus cámaras guardadas?"):
            return
        self.config["saved_cameras"] = [c for c in saved_list if c.get("id") != cam_id]
        self._guardar_config()
        self._renderizar_lista_camaras_guardadas()
        self._actualizar_combos_camaras_guardadas()
        self._registrar_evento("INFO", "Cámara Eliminada", f"Se borró '{cam_obj.get('name')}' del registro.")

    def _toggle_modo_prueba_pc(self):
        """
        Enciende o apaga la cámara integrada de la PC (índice 0) ÚNICAMENTE para Modo de Prueba.
        Al desactivarlo, cierra inmediatamente la cámara de la laptop y vuelve a la IP configurada.
        """
        self.modo_prueba_pc = not self.modo_prueba_pc
        if self.modo_prueba_pc:
            if self.config.get("camera_url", "") != "0":
                self.ultima_ip_real = self.config["camera_url"]
            self.btn_test_pc_cam.configure(
                text="⏹ Desactivar Modo Prueba",
                bg=self.C_AMBER, fg="#090D16"
            )
            self.cam = self.cam_pool.reconectar("0", max_width=int(self.config.get("max_width", 640)))
            self._limpiar_pool_inactivas()
            self._registrar_evento("WARN", "Modo Prueba Activado", "Cámara de la PC encendida temporalmente para pruebas.")
        else:
            self.btn_test_pc_cam.configure(
                text="🧪 Modo Prueba: OFF",
                bg="#374151", fg="#D1D5DB"
            )
            url_restaurar = self.ultima_ip_real or self.config.get("camera_url", "http://10.66.172.56:8080/video")
            if url_restaurar == "0":
                url_restaurar = "http://10.66.172.56:8080/video"
            self.config["camera_url"] = url_restaurar
            self._guardar_config()
            user, pwd = self._buscar_credenciales_guardadas(url_restaurar)
            self.cam = self.cam_pool.reconectar(url_restaurar, max_width=int(self.config.get("max_width", 640)), username=user, password=pwd)
            self._limpiar_pool_inactivas()
            self._registrar_evento("INFO", "Modo Prueba Finalizado", f"Cámara de la PC apagada. Reconectando a {url_restaurar}")

    def _activar_camara_como_principal(self, url_cam, nombre_cam=""):
        url_str = str(url_cam).strip()
        if url_str == "0":
            # Si pidió explícitamente "0", activar modo prueba temporal sin sobrescribir su IP real
            if not self.modo_prueba_pc:
                self._toggle_modo_prueba_pc()
            return

        if self.modo_prueba_pc:
            self.modo_prueba_pc = False
            if hasattr(self, "btn_test_pc_cam"):
                self.btn_test_pc_cam.configure(
                    text="🧪 Modo Prueba: OFF",
                    bg="#374151", fg="#D1D5DB"
                )

        self.ultima_ip_real = url_str
        self.config["camera_url"] = url_str
        self._guardar_config()

        self.ent_quick_ip.delete(0, tk.END)
        self.ent_quick_ip.insert(0, url_str)
        if hasattr(self, "ent_cam_url"):
            self.ent_cam_url.delete(0, tk.END)
            self.ent_cam_url.insert(0, url_str)

        user, pwd = self._buscar_credenciales_guardadas(url_str)
        self.cam = self.cam_pool.reconectar(
            url_str,
            max_width=int(self.config.get("max_width", 640)),
            username=user,
            password=pwd
        )
        self._limpiar_pool_inactivas()
        self._actualizar_combos_camaras_guardadas()
        self._registrar_evento("INFO", "Cámara Activa", f"Conectada cámara principal: {nombre_cam or url_str}")

    def _enviar_camara_a_quad_disponible(self, url_cam, nombre_cam=""):
        qc = self.config.setdefault("quad_cameras", ["-- Desactivada --", "-- Desactivada --", "-- Desactivada --", "-- Desactivada --"])
        # Buscar primer slot desactivado, o si ya está asignada no duplicar
        if url_cam not in qc:
            colocada = False
            for i in range(4):
                if qc[i] in ("-- Desactivada --", "", None):
                    qc[i] = url_cam
                    colocada = True
                    break
            if not colocada:
                qc[3] = url_cam
        self._guardar_config()
        self._sincronizar_streams_quad()
        self._actualizar_combos_camaras_guardadas()
        self._cambiar_vista("multicam")

    def _renderizar_lista_camaras_guardadas(self):
        for w in self.saved_list_inner.winfo_children():
            w.destroy()

        saved_list = self.config.get("saved_cameras", [])
        if not saved_list:
            tk.Label(
                self.saved_list_inner,
                text="Aún no hay cámaras guardadas.\nUsa el formulario de arriba o el escáner para almacenar tus cámaras.",
                font=("Segoe UI", 9), bg="#0B1320", fg=self.C_MUTED, pady=24
            ).pack()
            return

        cam_actual = str(self.config.get("camera_url", "")).strip()
        for c in saved_list:
            es_activa = (str(c.get("url", "")).strip() == cam_actual)
            card = tk.Frame(
                self.saved_list_inner,
                bg="#1E293B" if not es_activa else "#172554",
                highlightbackground="#3B82F6" if es_activa else "#334155",
                highlightthickness=1,
                padx=10, pady=8
            )
            card.pack(fill="x", pady=4)

            top_r = tk.Frame(card, bg=card["bg"])
            top_r.pack(fill="x")

            icono = "💻" if c.get("type") == "usb" else ("📹" if c.get("type") == "rtsp" else "📱")
            candado = " 🔑" if c.get("username") else ""
            badge_act = "  [ACTIVA]" if es_activa else ""

            tk.Label(
                top_r, text=f"{icono} {c.get('name', 'Cámara')}{candado}{badge_act}",
                font=("Segoe UI", 9, "bold"), bg=card["bg"],
                fg="#60A5FA" if es_activa else self.C_TEXT
            ).pack(side="left")

            # Botones de acción (Conectar, 4x, Editar, Borrar)
            btns = tk.Frame(top_r, bg=card["bg"])
            btns.pack(side="right")

            tk.Button(
                btns, text="▶️ Usar", font=("Segoe UI", 8, "bold"),
                bg=self.C_GREEN, fg="white", bd=0, relief="flat", padx=8, pady=2, cursor="hand2",
                command=lambda u=c["url"], n=c["name"]: self._activar_camara_como_principal(u, n)
            ).pack(side="left", padx=2)

            tk.Button(
                btns, text="📺 +4x", font=("Segoe UI", 8, "bold"),
                bg="#2563EB", fg="white", bd=0, relief="flat", padx=7, pady=2, cursor="hand2",
                command=lambda u=c["url"], n=c["name"]: self._enviar_camara_a_quad_disponible(u, n)
            ).pack(side="left", padx=2)

            tk.Button(
                btns, text="✏️", font=("Segoe UI", 8),
                bg="#334155", fg="white", bd=0, relief="flat", padx=6, pady=2, cursor="hand2",
                command=lambda cam=c: self._cargar_camara_en_formulario(cam, es_edicion=True)
            ).pack(side="left", padx=2)

            tk.Button(
                btns, text="🗑️", font=("Segoe UI", 8),
                bg="#450A0A", fg="#FCA5A5", bd=0, relief="flat", padx=6, pady=2, cursor="hand2",
                command=lambda cid=c["id"]: self._eliminar_camara_guardada(cid)
            ).pack(side="left", padx=2)

            # Mostrar URL ocultando contraseña si tiene
            url_disp = str(c.get("url", ""))
            if c.get("password") and ":" in url_disp and "@" in url_disp:
                url_disp = url_disp.replace(f":{c['password']}@", ":****@")

            tk.Label(
                card, text=f"Dirección: {url_disp}",
                font=("Consolas", 8), bg=card["bg"], fg=self.C_MUTED
            ).pack(anchor="w", pady=(3, 0))

    def _actualizar_combos_camaras_guardadas(self):
        saved_list = self.config.get("saved_cameras", [])
        items_header = [f"{c['name']}  [{c['url']}]" for c in saved_list]
        self.combo_saved_header["values"] = items_header

        cam_actual = str(self.config.get("camera_url", "")).strip()
        sel_idx = -1
        for i, c in enumerate(saved_list):
            if str(c.get("url", "")).strip() == cam_actual:
                sel_idx = i
                break
        if sel_idx >= 0:
            self.combo_saved_header.current(sel_idx)
        else:
            self.combo_saved_header.set("Seleccionar guardada...")

        # Actualizar también los 4 combos del Monitor de 4 Cámaras
        items_quad = ["-- Desactivada --"] + [f"{c['name']}  [{c['url']}]" for c in saved_list]
        qc = self.config.get("quad_cameras", ["-- Desactivada --"] * 4)
        for idx, slot in enumerate(self.quad_slots_ui):
            cb = slot["combo"]
            cb["values"] = items_quad
            q_val = qc[idx] if idx < len(qc) else "-- Desactivada --"
            if q_val in ("-- Desactivada --", "", None):
                cb.current(0)
            else:
                encontrado = False
                for i, c in enumerate(saved_list):
                    if str(c.get("url", "")).strip() == str(q_val).strip():
                        cb.current(i + 1)
                        encontrado = True
                        break
                if not encontrado:
                    cb.set(str(q_val))

    def _on_header_saved_cam_selected(self, event=None):
        idx = self.combo_saved_header.current()
        saved_list = self.config.get("saved_cameras", [])
        if 0 <= idx < len(saved_list):
            cam = saved_list[idx]
            self._activar_camara_como_principal(cam["url"], cam["name"])

    # -----------------------------------------------------------------
    # ESCANEO DE RED Y WEBCAMS EN SEGUNDO PLANO
    # -----------------------------------------------------------------
    def _iniciar_escaneo_completo(self):
        self.btn_scan_net.configure(state="disabled", text="⏳ Escaneando...")
        self.lbl_scan_progress.configure(
            text="⏳ Iniciando escaneo de puertos USB (0..5), Wi-Fi, Ethernet y ONVIF...",
            fg=self.C_AMBER
        )
        for w in self.scan_list_inner.winfo_children():
            w.destroy()

        tk.Label(
            self.scan_list_inner,
            text="Escaneando Webcams USB y buscando cámaras IP/Wi-Fi/Ethernet en toda la red...\nPor favor espera unos segundos.",
            font=("Segoe UI", 9), bg="#0B1320", fg=self.C_MUTED, pady=35
        ).pack()

        subred_extra = self.ent_subred_extra.get().strip() if hasattr(self, "ent_subred_extra") else ""
        threading.Thread(
            target=self._worker_escaneo_completo,
            args=(subred_extra,),
            daemon=True
        ).start()

    def _worker_escaneo_completo(self, subred_extra):
        def informar(msg):
            self.root.after(0, lambda m=msg: self.lbl_scan_progress.configure(text=f"⏳ {m}", fg=self.C_AMBER))

        encontradas = EscanerRedYWebcams.escanear_todo(
            camera_pool=self.cam_pool,
            subred_extra=subred_extra,
            callback_progreso=informar
        )
        self.root.after(0, lambda: self._mostrar_resultados_escaneo_completo(encontradas))

    def _mostrar_resultados_escaneo_completo(self, encontradas):
        self.btn_scan_net.configure(state="normal", text="🔍 Escanear Ahora")
        self.ultimas_camaras_detectadas = encontradas

        for w in self.scan_list_inner.winfo_children():
            w.destroy()

        if not encontradas:
            self.lbl_scan_progress.configure(
                text="⚠️ No se detectaron cámaras nuevas. Revisa la conexión o agrégala manualmente a la derecha.",
                fg=self.C_AMBER
            )
            tk.Label(
                self.scan_list_inner,
                text="No se encontraron cámaras activas en el escaneo automático.\n\n"
                     "• Para Webcam USB: verifica que esté bien conectada al puerto USB.\n"
                     "• Para Celular: abre IP Webcam / DroidCam y activa 'Iniciar Servidor'.\n"
                     "• Para Cámara Ethernet/Wi-Fi en otra subred: escribe su IP en el formulario derecho.",
                font=("Segoe UI", 9), bg="#0B1320", fg=self.C_MUTED, justify="left", pady=20
            ).pack(padx=12)
            return

        self.lbl_scan_progress.configure(
            text=f"✅ ¡Escaneo completado! Se detectaron {len(encontradas)} cámara(s) en USB / Red:",
            fg=self.C_GREEN
        )

        # Auto-registrar y vincular credenciales guardadas si la IP ya existe en saved_cameras
        auto_conectada = False
        for cam in encontradas:
            guardada = next(
                (
                    c for c in self.config.get("saved_cameras", [])
                    if str(c.get("url")) == str(cam["url"]) or str(c.get("host")) == str(cam["host"])
                ),
                None
            )
            if guardada is not None:
                # Heredar usuario, contraseña y URL autenticada si ya estaba guardada
                cam["username"] = guardada.get("username", "")
                cam["password"] = guardada.get("password", "")
                cam["url"] = construir_url_camara(guardada)
                if cam["username"]:
                    cam["detalle"] = f"🟢 Detectada en red ({cam['host']}) · Credenciales guardadas listas ({cam['username']})"
            elif not cam.get("requires_auth"):
                nueva_entry = {
                    "id": f"cam_{uuid.uuid4().hex[:8]}",
                    "name": cam["name"],
                    "type": cam["type"],
                    "host": cam["host"],
                    "port": cam["port"],
                    "username": "",
                    "password": "",
                    "stream_path": cam["stream_path"],
                    "url": cam["url"],
                }
                self.config["saved_cameras"].append(nueva_entry)

            tiene_acceso = (not cam.get("requires_auth")) or bool(cam.get("username"))
            if (
                self.config.get("auto_connect_open_cams", True)
                and tiene_acceso
                and not auto_conectada
                and (not self.cam or not self.cam.connected or cam["type"] != "usb")
            ):
                auto_conectada = True
                self._activar_camara_como_principal(cam["url"], cam["name"])

        self._guardar_config()
        self._renderizar_lista_camaras_guardadas()
        self._actualizar_combos_camaras_guardadas()

        # Renderizar tarjetas de cámaras encontradas
        for cam in encontradas:
            pide_clave = bool(cam.get("requires_auth")) and not bool(cam.get("username"))
            borde_col = self.C_AMBER if pide_clave else "#059669"

            card = tk.Frame(
                self.scan_list_inner, bg="#1E293B",
                highlightbackground=borde_col, highlightthickness=1,
                padx=10, pady=8
            )
            card.pack(fill="x", pady=4)

            top_r = tk.Frame(card, bg="#1E293B")
            top_r.pack(fill="x")

            tk.Label(
                top_r, text=f"{cam['icono']}  {cam['name']}",
                font=("Segoe UI", 9, "bold"), bg="#1E293B", fg=self.C_TEXT
            ).pack(side="left")

            btns = tk.Frame(top_r, bg="#1E293B")
            btns.pack(side="right")

            tk.Button(
                btns, text="⚡ Conectar Ahora", font=("Segoe UI", 8, "bold"),
                bg=self.C_GREEN, fg="white", bd=0, relief="flat", padx=10, pady=3, cursor="hand2",
                command=lambda c=cam: self._activar_camara_como_principal(c["url"], c["name"])
            ).pack(side="left", padx=2)

            tk.Button(
                btns, text="🔑 Usuario/Clave" if cam.get("requires_auth") else "✏️ Editar",
                font=("Segoe UI", 8, "bold"),
                bg=self.C_AMBER if pide_clave else "#334155",
                fg="#090D16" if pide_clave else "white",
                bd=0, relief="flat", padx=9, pady=3, cursor="hand2",
                command=lambda c=cam: self._cargar_camara_en_formulario(c, es_edicion=False)
            ).pack(side="left", padx=2)

            tk.Button(
                btns, text="📺 +4x", font=("Segoe UI", 8, "bold"),
                bg=self.C_BLUE, fg="white", bd=0, relief="flat", padx=7, pady=3, cursor="hand2",
                command=lambda c=cam: self._enviar_camara_a_quad_disponible(c["url"], c["name"])
            ).pack(side="left", padx=2)

            tk.Label(
                card, text=cam["detalle"],
                font=("Consolas", 8), bg="#1E293B",
                fg="#FCD34D" if pide_clave else "#6EE7B7"
            ).pack(anchor="w", pady=(3, 0))

    # -----------------------------------------------------------------
    # LÓGICA DEL MONITOR DE 4 CÁMARAS SIMULTÁNEAS (2x2)
    # -----------------------------------------------------------------
    def _sincronizar_streams_quad(self):
        qc = self.config.get("quad_cameras", ["-- Desactivada --"] * 4)
        for i in range(4):
            q_url = qc[i] if i < len(qc) else "-- Desactivada --"
            if q_url and q_url not in ("-- Desactivada --", "", None):
                user, pwd = self._buscar_credenciales_guardadas(q_url)
                self.cam_pool.obtener_o_crear(q_url, max_width=480, username=user, password=pwd)
        self._limpiar_pool_inactivas()

    def _on_quad_cam_selected(self, slot_idx):
        slot = self.quad_slots_ui[slot_idx]
        sel_idx = slot["combo"].current()
        saved_list = self.config.get("saved_cameras", [])

        if sel_idx <= 0:
            self.config["quad_cameras"][slot_idx] = "-- Desactivada --"
        elif 1 <= sel_idx <= len(saved_list):
            cam = saved_list[sel_idx - 1]
            self.config["quad_cameras"][slot_idx] = cam["url"]

        slot["last_frame_id"] = -1
        self._guardar_config()
        self._sincronizar_streams_quad()

    def _apagar_slot_quad(self, slot_idx):
        self.config["quad_cameras"][slot_idx] = "-- Desactivada --"
        self.quad_slots_ui[slot_idx]["combo"].current(0)
        self.quad_slots_ui[slot_idx]["last_frame_id"] = -1
        self._guardar_config()
        self._sincronizar_streams_quad()

    def _reconectar_slot_quad(self, slot_idx):
        q_url = self.config["quad_cameras"][slot_idx]
        if q_url and q_url != "-- Desactivada --":
            user, pwd = self._buscar_credenciales_guardadas(q_url)
            self.cam_pool.reconectar(q_url, max_width=480, username=user, password=pwd)

    def _enviar_slot_a_principal(self, slot_idx):
        q_url = self.config["quad_cameras"][slot_idx]
        if not q_url or q_url == "-- Desactivada --":
            return
        self._activar_camara_como_principal(q_url, f"Cámara Canal {slot_idx + 1}")
        self._cambiar_vista("monitoreo")

    def _auto_asignar_4_camaras(self):
        saved_list = self.config.get("saved_cameras", [])
        urls = [c["url"] for c in saved_list[:4]]
        while len(urls) < 4:
            urls.append("-- Desactivada --")
        self.config["quad_cameras"] = urls
        self._guardar_config()
        self._sincronizar_streams_quad()
        self._actualizar_combos_camaras_guardadas()

    def _toggle_quad_ai(self):
        self.quad_ai_enabled = not self.quad_ai_enabled
        if self.quad_ai_enabled:
            self.btn_quad_ai.configure(
                text="🤖 Modo: Detección IA Activa en 4 Cámaras (Round-Robin)",
                bg="#065F46", fg="#6EE7B7"
            )
        else:
            self.btn_quad_ai.configure(
                text="👁️ Modo: Solo Visualización (Ultra-Ligero)",
                bg="#1E293B", fg="#38BDF8"
            )

    def _actualizar_vista_multicam(self):
        """
        Renderiza los 4 canales en una cuadrícula 2x2 ultra-ligera.
        Solo redibuja un canal cuando su cámara tiene un frame nuevo, evitando
        consumir CPU o congelar la interfaz.
        """
        qc = self.config.get("quad_cameras", ["-- Desactivada --"] * 4)
        conf_umbral = self.config["confidence"] / 100.0

        # Si IA en 4 cámaras está activa, turnar 1 cámara por ciclo al hilo YOLO
        if self.quad_ai_enabled:
            activas_idx = [i for i in range(4) if qc[i] and qc[i] != "-- Desactivada --"]
            if activas_idx:
                self.quad_rr_index = (self.quad_rr_index + 1) % len(activas_idx)
                idx_turno = activas_idx[self.quad_rr_index]
                cam_t = self.cam_pool.obtener_o_crear(qc[idx_turno], max_width=480)
                if cam_t and cam_t.connected:
                    frm_t, fid_t = cam_t.read()
                    if frm_t is not None:
                        self.yolo_engine.enviar_frame(frm_t, fid_t, source_key=f"quad_{idx_turno}")

        for idx in range(4):
            slot = self.quad_slots_ui[idx]
            cv_q = slot["canvas"]
            q_url = qc[idx] if idx < len(qc) else "-- Desactivada --"

            cw = max(160, cv_q.winfo_width())
            ch = max(120, cv_q.winfo_height())

            if not q_url or q_url == "-- Desactivada --":
                slot["lbl_status"].configure(text="○ Desactivada", fg=self.C_MUTED)
                cv_q.delete("all")
                cv_q.create_text(
                    cw // 2, ch // 2,
                    text=f"CANAL {idx + 1} DESACTIVADO\nSelecciona una cámara arriba",
                    fill="#4B5563", font=("Segoe UI", 10, "bold"), justify="center"
                )
                continue

            user, pwd = self._buscar_credenciales_guardadas(q_url)
            cam_obj = self.cam_pool.obtener_o_crear(q_url, max_width=480, username=user, password=pwd)
            if cam_obj is None or not cam_obj.connected:
                msg = cam_obj.status_msg if cam_obj else "Sin conexión"
                slot["lbl_status"].configure(text=f"⏳ {msg[:26]}", fg=self.C_AMBER)
                cv_q.delete("all")
                cv_q.create_text(
                    cw // 2, ch // 2,
                    text=f"CONECTANDO CANAL {idx + 1}...\n{q_url}\n{msg}",
                    fill="#F59E0B", font=("Consolas", 9), justify="center"
                )
                continue

            frm_raw, fid = cam_obj.read()
            if frm_raw is None:
                continue

            slot["lbl_status"].configure(text=f"● EN VIVO ({cam_obj.fps_cam:.0f} FPS)", fg=self.C_GREEN)

            # Solo redibujar si llegó un frame nuevo o cambió el tamaño del canvas
            if fid == slot["last_frame_id"] and hasattr(cv_q, "img_tk"):
                continue
            slot["last_frame_id"] = fid

            frm = frm_raw.copy() if self.quad_ai_enabled else frm_raw
            if self.quad_ai_enabled:
                dets, _ = self.yolo_engine.obtener_detecciones(source_key=f"quad_{idx}")
                for (x1, y1, x2, y2, cf) in dets:
                    if cf >= conf_umbral:
                        cv2.rectangle(frm, (x1, y1), (x2, y2), (129, 185, 16), 2)
                        cv2.putText(frm, f"Persona {cf*100:.0f}%", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (129, 185, 16), 1)

            fh, fw = frm.shape[:2]
            scale = min(cw / float(fw), ch / float(fh))
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            x0, y0 = (cw - nw) // 2, (ch - nh) // 2

            resized = cv2.resize(frm, (nw, nh), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            cv_q.img_tk = img_tk
            cv_q.delete("all")
            cv_q.create_image(x0, y0, anchor="nw", image=img_tk)

    # -----------------------------------------------------------------
    # GESTIÓN DE MÚLTIPLES GEOCERCAS Y ACCIONES POR ÁREA (ALARMA vs CORTE)
    # -----------------------------------------------------------------
    @staticmethod
    def _obtener_accion_zona(gz):
        act = str(gz.get("action", "")).lower()
        if act in ("alarma", "corte", "seguro"):
            return act
        return "seguro" if gz.get("inverted", False) else "corte"

    @classmethod
    def _info_visual_accion(cls, accion):
        if accion == "alarma":
            return "🔔", "1: Encender Alarma (Relé 1)", "#78350F", "#FCD34D", "#92400E"
        if accion == "seguro":
            return "🛡️", "Zona Segura", "#065F46", "#6EE7B7", "#047857"
        return "⚡", "2: Cortar Corriente (Relé 2)", "#7F1D1D", "#FCA5A5", "#991B1B"

    def _refrescar_pills_zonas(self):
        geofences = self.config.get("geofences", [])
        if self.selected_zone_idx >= len(geofences):
            self.selected_zone_idx = max(0, len(geofences) - 1)

        items_combo = []
        for idx, gz in enumerate(geofences):
            acc = self._obtener_accion_zona(gz)
            icono, etiqueta_tipo, _, _, _ = self._info_visual_accion(acc)
            items_combo.append(f"{icono} {gz['name']} · {etiqueta_tipo}")

        if hasattr(self, "combo_zonas"):
            self.combo_zonas["values"] = items_combo
            if items_combo and 0 <= self.selected_zone_idx < len(items_combo):
                self.combo_zonas.current(self.selected_zone_idx)

        # Actualizar botón de acción de la zona seleccionada
        if hasattr(self, "btn_zone_safe_toggle") and 0 <= self.selected_zone_idx < len(geofences):
            acc = self._obtener_accion_zona(geofences[self.selected_zone_idx])
            icono, etiqueta_tipo, bg_c, fg_c, act_bg = self._info_visual_accion(acc)
            self.btn_zone_safe_toggle.configure(
                text=f"{icono} {etiqueta_tipo}",
                bg=bg_c, fg=fg_c, activebackground=act_bg
            )

    def _on_combo_zona_selected(self, event=None):
        if hasattr(self, "combo_zonas"):
            idx = self.combo_zonas.current()
            if idx >= 0:
                self._seleccionar_geocerca(idx)

    def _seleccionar_geocerca(self, idx):
        self.selected_zone_idx = idx
        self._refrescar_pills_zonas()

    def _agregar_nueva_geocerca(self):
        geofences = self.config["geofences"]
        nuevo_num = len(geofences) + 1
        offset = ((nuevo_num - 1) % 5) * 0.05
        x1 = max(0.08, min(0.55, 0.16 + offset))
        y1 = max(0.10, min(0.55, 0.18 + offset))
        x2 = min(0.92, x1 + 0.28)
        y2 = min(0.90, y1 + 0.32)
        accion_def = "alarma" if nuevo_num == 1 else "corte"
        nueva = {
            "name": f"Zona {nuevo_num}",
            "action": accion_def,
            "inverted": False,
            "points": [
                [round(x1, 3), round(y1, 3)],
                [round(x2, 3), round(y1, 3)],
                [round(x2, 3), round(y2, 3)],
                [round(x1, 3), round(y2, 3)],
            ]
        }
        geofences.append(nueva)
        self.selected_zone_idx = len(geofences) - 1
        self.config["zone_mode"] = "geocerca_poly"
        self._guardar_config()
        self._refrescar_botones_modo()
        self._refrescar_pills_zonas()
        _, desc_acc, _, _, _ = self._info_visual_accion(accion_def)
        self._registrar_evento("INFO", "Nueva Geocerca", f"Se creó {nueva['name']} ({desc_acc}).")

    def _eliminar_geocerca_actual(self):
        self._eliminar_geocerca_idx(self.selected_zone_idx)

    def _eliminar_geocerca_idx(self, idx):
        geofences = self.config["geofences"]
        if len(geofences) <= 1:
            messagebox.showinfo("Aviso", "Debe existir al menos 1 geocerca activa.")
            return
        if not (0 <= idx < len(geofences)):
            return
        eliminada = geofences.pop(idx)
        for i, gz in enumerate(geofences):
            gz["name"] = f"Zona {i + 1}"
        self.selected_zone_idx = min(max(0, idx), len(geofences) - 1)
        self._guardar_config()
        self._refrescar_pills_zonas()
        self._registrar_evento("INFO", "Geocerca Eliminada", f"Se eliminó {eliminada['name']}.")

    def _alternar_inversion_zona(self, idx):
        geofences = self.config.get("geofences", [])
        if not (0 <= idx < len(geofences)):
            return
        gz = geofences[idx]
        actual = self._obtener_accion_zona(gz)
        orden = ["alarma", "corte", "seguro"]
        nueva_acc = orden[(orden.index(actual) + 1) % len(orden)]
        self._set_accion_zona(idx, nueva_acc)

    def _set_accion_zona(self, idx, nueva_accion):
        geofences = self.config.get("geofences", [])
        if not (0 <= idx < len(geofences)):
            return
        gz = geofences[idx]
        gz["action"] = nueva_accion
        gz["inverted"] = (nueva_accion == "seguro")
        self.selected_zone_idx = idx
        self._guardar_config()
        self._refrescar_pills_zonas()
        icono, estado_txt, _, fg_col, _ = self._info_visual_accion(nueva_accion)
        self.lbl_poly_hint.configure(
            text=f"{icono} {gz['name']}: {estado_txt}",
            fg=fg_col
        )
        self._registrar_evento("INFO", "Acción de Zona", f"{gz['name']} configurada como: {estado_txt}")

    def _set_inversion_zona(self, idx, es_segura):
        self._set_accion_zona(idx, "seguro" if es_segura else "corte")

    def _copiar_geocerca(self, idx=None):
        geofences = self.config.get("geofences", [])
        if idx is None:
            idx = self.selected_zone_idx
        if not (0 <= idx < len(geofences)):
            return
        gz = geofences[idx]
        self.zone_clipboard = {
            "source_name": gz.get("name", f"Zona {idx + 1}"),
            "action": self._obtener_accion_zona(gz),
            "inverted": bool(gz.get("inverted", False)),
            "points": [[float(p[0]), float(p[1])] for p in gz.get("points", [])]
        }
        self.lbl_poly_hint.configure(
            text=f"📋 {gz['name']} copiada",
            fg="#93C5FD"
        )
        self._registrar_evento("INFO", "Zona Copiada", f"Se copió {gz['name']} al portapapeles.")

    def _pegar_geocerca(self, target_norm=None):
        if not self.zone_clipboard or not self.zone_clipboard.get("points"):
            messagebox.showinfo("Portapapeles vacío", "Primero copia una zona.")
            return
        geofences = self.config["geofences"]
        src_pts = self.zone_clipboard["points"]
        cx = sum(p[0] for p in src_pts) / len(src_pts)
        cy = sum(p[1] for p in src_pts) / len(src_pts)

        if target_norm is not None:
            dx = target_norm[0] - cx
            dy = target_norm[1] - cy
        else:
            dx = 0.05
            dy = 0.05

        xs = [p[0] + dx for p in src_pts]
        ys = [p[1] + dy for p in src_pts]
        if min(xs) < 0.02:
            dx += (0.02 - min(xs))
        if max(xs) > 0.98:
            dx -= (max(xs) - 0.98)
        if min(ys) < 0.02:
            dy += (0.02 - min(ys))
        if max(ys) > 0.98:
            dy -= (max(ys) - 0.98)

        nuevos_pts = [[round(p[0] + dx, 4), round(p[1] + dy, 4)] for p in src_pts]
        nuevo_num = len(geofences) + 1
        acc_clip = self.zone_clipboard.get("action", "corte")
        nueva = {
            "name": f"Zona {nuevo_num}",
            "action": acc_clip,
            "inverted": (acc_clip == "seguro"),
            "points": nuevos_pts
        }
        geofences.append(nueva)
        self.selected_zone_idx = len(geofences) - 1
        self.config["zone_mode"] = "geocerca_poly"
        self._guardar_config()
        self._refrescar_botones_modo()
        self._refrescar_pills_zonas()
        self.lbl_poly_hint.configure(
            text=f"📌 {nueva['name']} pegada",
            fg="#6EE7B7"
        )
        self._registrar_evento("INFO", "Zona Pegada", f"Se creó {nueva['name']}.")

    def _resetear_zona_a_4_puntos(self, idx=None):
        geofences = self.config["geofences"]
        if not geofences:
            return
        if idx is None:
            idx = self.selected_zone_idx
        if not (0 <= idx < len(geofences)):
            return
        gz = geofences[idx]
        pts = gz["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0.08, min(xs)), min(0.92, max(xs))
        y1, y2 = max(0.08, min(ys)), min(0.92, max(ys))
        if x2 - x1 < 0.12:
            x2 = min(0.95, x1 + 0.25)
        if y2 - y1 < 0.12:
            y2 = min(0.95, y1 + 0.25)
        gz["points"] = [
            [round(x1, 3), round(y1, 3)],
            [round(x2, 3), round(y1, 3)],
            [round(x2, 3), round(y2, 3)],
            [round(x1, 3), round(y2, 3)],
        ]
        self._guardar_config()
        self._refrescar_pills_zonas()

    def _toggle_modo_agregar_puntos(self):
        self.modo_agregar_puntos = not self.modo_agregar_puntos
        if self.modo_agregar_puntos:
            self.btn_add_pts.configure(
                text="✏️ +Puntos: ON",
                bg=self.C_AMBER, fg="#090D16"
            )
            self.lbl_poly_hint.configure(
                text="Haz clic en el video para añadir punto",
                fg="#FCD34D"
            )
        else:
            self.btn_add_pts.configure(
                text="✏️ +Puntos",
                bg="#1E293B", fg="#E5E7EB"
            )
            self.lbl_poly_hint.configure(
                text="",
                fg="#9CA3AF"
            )

    @staticmethod
    def _insertar_punto_en_arista_mas_cercana(points, nx, ny):
        n = len(points)
        if n < 2:
            points.append([round(nx, 4), round(ny, 4)])
            return len(points) - 1

        mejor_idx = n
        menor_dist = float("inf")
        p = np.array([nx, ny], dtype=float)

        for i in range(n):
            a = np.array(points[i], dtype=float)
            b = np.array(points[(i + 1) % n], dtype=float)
            ab = b - a
            denom = np.dot(ab, ab)
            if denom < 1e-9:
                proy = a
            else:
                t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
                proy = a + t * ab
            dist = float(np.linalg.norm(p - proy))
            if dist < menor_dist:
                menor_dist = dist
                mejor_idx = i + 1

        points.insert(mejor_idx, [round(nx, 4), round(ny, 4)])
        return mejor_idx

    # -----------------------------------------------------------------
    # INTERACCIÓN CON EL MOUSE SOBRE EL VIDEO (ARRASTRAR, AÑADIR, BORRAR)
    # -----------------------------------------------------------------
    def _canvas_to_norm(self, cx, cy):
        if self.vid_w <= 1 or self.vid_h <= 1:
            return 0.5, 0.5
        nx = (cx - self.vid_x0) / float(self.vid_w)
        ny = (cy - self.vid_y0) / float(self.vid_h)
        return max(0.01, min(0.99, nx)), max(0.01, min(0.99, ny))

    def _on_canvas_resize(self, event):
        self.canvas_w = max(100, event.width)
        self.canvas_h = max(100, event.height)

    def _on_canvas_hover(self, event):
        if self.config["zone_mode"] != "geocerca_poly":
            self.hover_info = None
            return
        nx, ny = self._canvas_to_norm(event.x, event.y)
        geofences = self.config["geofences"]
        tol_v = 0.028

        for z_idx, gz in enumerate(geofences):
            for pt_idx, (px, py) in enumerate(gz["points"]):
                if abs(nx - px) < tol_v and abs(ny - py) < tol_v:
                    self.hover_info = ("vertex", z_idx, pt_idx)
                    self.video_canvas.configure(cursor="hand2")
                    return

        if 0 <= self.selected_zone_idx < len(geofences):
            pts = geofences[self.selected_zone_idx]["points"]
            n = len(pts)
            for i in range(n):
                mx = (pts[i][0] + pts[(i + 1) % n][0]) * 0.5
                my = (pts[i][1] + pts[(i + 1) % n][1]) * 0.5
                if abs(nx - mx) < 0.024 and abs(ny - my) < 0.024:
                    self.hover_info = ("edge", self.selected_zone_idx, i, mx, my)
                    self.video_canvas.configure(cursor="plus")
                    return

        self.hover_info = None
        self.video_canvas.configure(cursor="tcross" if self.modo_agregar_puntos else "crosshair")

    def _on_canvas_click(self, event):
        nx, ny = self._canvas_to_norm(event.x, event.y)
        modo = self.config["zone_mode"]

        if modo == "linea_vertical":
            self.dragging_mode = "line_v"
            self.config["line_x"] = round(nx, 4)
            return

        if modo == "linea_horizontal":
            self.dragging_mode = "line_h"
            self.config["line_y"] = round(ny, 4)
            return

        geofences = self.config["geofences"]
        tol_v = 0.030

        for z_idx, gz in enumerate(geofences):
            for pt_idx, (px, py) in enumerate(gz["points"]):
                if abs(nx - px) < tol_v and abs(ny - py) < tol_v:
                    self.selected_zone_idx = z_idx
                    self.dragging_mode = "vertex"
                    self.drag_zone_idx = z_idx
                    self.drag_vertex_idx = pt_idx
                    self._refrescar_pills_zonas()
                    return

        if self.modo_agregar_puntos and 0 <= self.selected_zone_idx < len(geofences):
            pts = geofences[self.selected_zone_idx]["points"]
            new_idx = self._insertar_punto_en_arista_mas_cercana(pts, nx, ny)
            self.dragging_mode = "vertex"
            self.drag_zone_idx = self.selected_zone_idx
            self.drag_vertex_idx = new_idx
            self._refrescar_pills_zonas()
            return

        if 0 <= self.selected_zone_idx < len(geofences):
            pts = geofences[self.selected_zone_idx]["points"]
            n = len(pts)
            for i in range(n):
                mx = (pts[i][0] + pts[(i + 1) % n][0]) * 0.5
                my = (pts[i][1] + pts[(i + 1) % n][1]) * 0.5
                if abs(nx - mx) < 0.026 and abs(ny - my) < 0.026:
                    pts.insert(i + 1, [round(nx, 4), round(ny, 4)])
                    self.dragging_mode = "vertex"
                    self.drag_zone_idx = self.selected_zone_idx
                    self.drag_vertex_idx = i + 1
                    self._refrescar_pills_zonas()
                    return

        for z_idx, gz in enumerate(geofences):
            poly_np = np.array([[p[0] * 1000.0, p[1] * 1000.0] for p in gz["points"]], dtype=np.float32)
            if cv2.pointPolygonTest(poly_np, (nx * 1000.0, ny * 1000.0), False) >= 0:
                self.selected_zone_idx = z_idx
                self.dragging_mode = "zone_move"
                self.drag_zone_idx = z_idx
                self.drag_start_norm = (nx, ny)
                self.drag_orig_points = [[p[0], p[1]] for p in gz["points"]]
                self._refrescar_pills_zonas()
                return

    def _on_canvas_double_click(self, event):
        if self.config["zone_mode"] != "geocerca_poly":
            return
        nx, ny = self._canvas_to_norm(event.x, event.y)
        geofences = self.config["geofences"]
        if 0 <= self.selected_zone_idx < len(geofences):
            pts = geofences[self.selected_zone_idx]["points"]
            self._insertar_punto_en_arista_mas_cercana(pts, nx, ny)
            self._guardar_config()
            self._refrescar_pills_zonas()

    def _eliminar_vertice_zona(self, z_idx, pt_idx):
        geofences = self.config.get("geofences", [])
        if not (0 <= z_idx < len(geofences)):
            return
        pts = geofences[z_idx]["points"]
        if len(pts) <= 3:
            messagebox.showinfo("SafeVision", "Una geocerca necesita mínimo 3 puntos para formar un área cerrada.")
            return
        if 0 <= pt_idx < len(pts):
            pts.pop(pt_idx)
            self.selected_zone_idx = z_idx
            self._guardar_config()
            self._refrescar_pills_zonas()

    def _duplicar_geocerca(self, idx):
        self._copiar_geocerca(idx)
        self._pegar_geocerca(target_norm=None)

    def _mostrar_menu_contextual_zona(self, event, z_idx, pt_idx=None, click_norm=None):
        geofences = self.config.get("geofences", [])
        if not geofences:
            return
        if not (0 <= z_idx < len(geofences)):
            z_idx = self.selected_zone_idx
        self.selected_zone_idx = z_idx
        self._refrescar_pills_zonas()

        gz = geofences[z_idx]
        acc = self._obtener_accion_zona(gz)
        icono_a, estado_str, _, _, _ = self._info_visual_accion(acc)

        menu = tk.Menu(
            self.root, tearoff=0,
            bg="#0F172A", fg="#F8FAFC",
            activebackground="#2563EB", activeforeground="white",
            font=("Segoe UI", 9)
        )

        menu.add_command(
            label=f"📍 {gz['name']} · {icono_a} {estado_str}",
            state="disabled"
        )
        menu.add_separator()

        # Configuración de Acción por Área (1: Encender Alarma, 2: Cortar Corriente, 3: Zona Segura)
        menu.add_command(
            label=f"{'✓ ' if acc == 'alarma' else '    '}🔔 Acción 1: Encender Alarma (Relé 1)",
            command=lambda: self._set_accion_zona(z_idx, "alarma")
        )
        menu.add_command(
            label=f"{'✓ ' if acc == 'corte' else '    '}⚡ Acción 2: Cortar Corriente (Relé 2 + Alarma)",
            command=lambda: self._set_accion_zona(z_idx, "corte")
        )
        menu.add_command(
            label=f"{'✓ ' if acc == 'seguro' else '    '}🛡️ Configurar como: Zona Segura",
            command=lambda: self._set_accion_zona(z_idx, "seguro")
        )
        menu.add_command(
            label="⇄ Cambiar Acción de Zona",
            command=lambda: self._alternar_inversion_zona(z_idx)
        )
        menu.add_separator()

        # Copiar, Pegar y Duplicar
        menu.add_command(
            label=f"📋 Copiar {gz['name']}",
            command=lambda: self._copiar_geocerca(z_idx)
        )
        tiene_portapapeles = bool(self.zone_clipboard and self.zone_clipboard.get("points"))
        lbl_pegar = "📌 Pegar Zona Aquí" if (tiene_portapapeles and click_norm is not None) else "📌 Pegar Zona"
        menu.add_command(
            label=lbl_pegar,
            state="normal" if tiene_portapapeles else "disabled",
            command=lambda: self._pegar_geocerca(target_norm=click_norm)
        )
        menu.add_command(
            label=f"📑 Duplicar {gz['name']}",
            command=lambda: self._duplicar_geocerca(z_idx)
        )
        menu.add_separator()

        # Opciones de puntos y eliminación
        if pt_idx is not None:
            menu.add_command(
                label=f"❌ Eliminar Punto #{pt_idx + 1}",
                command=lambda: self._eliminar_vertice_zona(z_idx, pt_idx)
            )
        menu.add_command(
            label="▢ Reiniciar a 4 Puntos",
            command=lambda: self._resetear_zona_a_4_puntos(z_idx)
        )
        menu.add_command(
            label=f"🗑️ Eliminar {gz['name']}",
            command=lambda: self._eliminar_geocerca_idx(z_idx)
        )

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _on_canvas_right_click(self, event):
        if self.config["zone_mode"] != "geocerca_poly":
            return
        nx, ny = self._canvas_to_norm(event.x, event.y)
        geofences = self.config["geofences"]
        tol_v = 0.032

        # 1) Verificar si hizo clic derecho sobre un vértice
        for z_idx, gz in enumerate(geofences):
            pts = gz["points"]
            for pt_idx, (px, py) in enumerate(pts):
                if abs(nx - px) < tol_v and abs(ny - py) < tol_v:
                    self._mostrar_menu_contextual_zona(event, z_idx, pt_idx=pt_idx, click_norm=(nx, ny))
                    return

        # 2) Verificar si hizo clic derecho dentro de alguna geocerca
        for z_idx, gz in enumerate(geofences):
            pts = gz.get("points", [])
            if len(pts) >= 3:
                poly_np = np.array([[p[0] * 1000.0, p[1] * 1000.0] for p in pts], dtype=np.float32)
                if cv2.pointPolygonTest(poly_np, (nx * 1000.0, ny * 1000.0), False) >= 0:
                    self._mostrar_menu_contextual_zona(event, z_idx, pt_idx=None, click_norm=(nx, ny))
                    return

        # 3) Si hizo clic derecho fuera de los polígonos, mostrar el menú para la zona activa (permite Pegar Aquí)
        self._mostrar_menu_contextual_zona(event, self.selected_zone_idx, pt_idx=None, click_norm=(nx, ny))

    def _on_canvas_drag(self, event):
        if not self.dragging_mode:
            return
        nx, ny = self._canvas_to_norm(event.x, event.y)

        if self.dragging_mode == "line_v":
            self.config["line_x"] = round(nx, 4)
        elif self.dragging_mode == "line_h":
            self.config["line_y"] = round(ny, 4)
        elif self.dragging_mode == "vertex":
            pts = self.config["geofences"][self.drag_zone_idx]["points"]
            if 0 <= self.drag_vertex_idx < len(pts):
                pts[self.drag_vertex_idx][0] = round(nx, 4)
                pts[self.drag_vertex_idx][1] = round(ny, 4)
        elif self.dragging_mode == "zone_move":
            dx = nx - self.drag_start_norm[0]
            dy = ny - self.drag_start_norm[1]
            pts = self.config["geofences"][self.drag_zone_idx]["points"]
            xs = [p[0] + dx for p in self.drag_orig_points]
            ys = [p[1] + dy for p in self.drag_orig_points]
            if min(xs) < 0.01:
                dx += (0.01 - min(xs))
            if max(xs) > 0.99:
                dx -= (max(xs) - 0.99)
            if min(ys) < 0.01:
                dy += (0.01 - min(ys))
            if max(ys) > 0.99:
                dy -= (max(ys) - 0.99)
            for i, orig in enumerate(self.drag_orig_points):
                pts[i][0] = round(orig[0] + dx, 4)
                pts[i][1] = round(orig[1] + dy, 4)

    def _on_canvas_release(self, event):
        if self.dragging_mode:
            self.dragging_mode = None
            self._guardar_config()

    # -----------------------------------------------------------------
    # CAMBIO DE MODOS, CONFIGURACIÓN ESP32 Y CONTROLES DE CÁMARA
    # -----------------------------------------------------------------
    def _refrescar_puertos_com_ui(self):
        if not hasattr(self, "combo_esp32_port"):
            return
        puertos = ControladorESP32Reles.escanear_puertos_com()
        opciones = ["AUTO (Detectar ESP32 automáticamente)"]
        for p in puertos:
            opciones.append(p["desc"])
        self.combo_esp32_port["values"] = opciones

        guardado = str(self.config.get("esp32_port", "AUTO")).strip().upper()
        if guardado == "AUTO" or not guardado:
            self.combo_esp32_port.current(0)
        else:
            sel = 0
            for idx, p in enumerate(puertos):
                if p["port"] == guardado:
                    sel = idx + 1
                    break
            if sel > 0:
                self.combo_esp32_port.current(sel)
            else:
                self.combo_esp32_port.current(0)

    def _on_esp32_port_selected(self, event=None):
        raw_port = self.combo_esp32_port.get().strip() if hasattr(self, "combo_esp32_port") else "AUTO"
        if raw_port.upper().startswith("AUTO") or not raw_port:
            port_clean = "AUTO"
        else:
            port_clean = raw_port.split("—")[0].split()[0].strip().upper()
        esp_low = bool(self.var_esp32_low.get()) if hasattr(self, "var_esp32_low") else True

        self.config["esp32_mode"] = "usb"
        self.config["esp32_port"] = port_clean
        self.config["esp32_active_low"] = esp_low
        self._guardar_config()
        self.esp32.actualizar_config(mode="usb", serial_port=port_clean, baudrate=115200, active_low=esp_low)

    def _detectar_y_reconectar_usb(self):
        ControladorESP32Reles._wmi_cache_ts = 0.0
        puertos = ControladorESP32Reles.escanear_puertos_com()
        self.config["esp32_mode"] = "usb"
        if puertos:
            mejor = puertos[0]["port"]
            self.config["esp32_port"] = mejor
            self._guardar_config()
            self._refrescar_puertos_com_ui()
            self.esp32.actualizar_config(
                mode="usb",
                serial_port=mejor,
                baudrate=115200,
                active_low=bool(self.var_esp32_low.get()) if hasattr(self, "var_esp32_low") else True
            )
            self._registrar_evento("INFO", "ESP32 USB Detectada", f"Conectando a {puertos[0]['desc']}")
        else:
            self.config["esp32_port"] = "AUTO"
            self._guardar_config()
            self._refrescar_puertos_com_ui()
            self.esp32.actualizar_config(
                mode="usb",
                serial_port="AUTO",
                baudrate=115200,
                active_low=bool(self.var_esp32_low.get()) if hasattr(self, "var_esp32_low") else True
            )
            messagebox.showwarning(
                "ESP32 USB",
                "No se detectó ningún puerto USB COM activo en Windows.\n\n"
                "1. Verifica que el cable USB de la ESP32 sea de DATOS (no solo carga).\n"
                "2. Cierra el Monitor Serial de Arduino IDE si lo tienes abierto."
            )

    def _probar_rele_desde_ui(self, num_rele):
        nombres = {
            1: "Relé 1 (Alarma / Sirena)",
            2: "Relé 2 (Corte de Corriente)",
            3: "Relé 3 (Luz Verde OK)",
            4: "Relé 4 (Auxiliar)",
        }
        self.esp32.probar_rele_manual(num_rele, encender=None, duracion_seg=4.0)
        estado_act = "ACTIVADO" if self.esp32.relay_states[num_rele - 1] else "APAGADO"
        self._registrar_evento(
            "INFO",
            f"Prueba Manual Relé {num_rele}",
            f"{nombres.get(num_rele, f'Relé {num_rele}')} -> {estado_act} (por 4 seg)."
        )

    def _cambiar_modo_zona(self, nuevo_modo):
        self.config["zone_mode"] = nuevo_modo
        self._guardar_config()
        self._refrescar_botones_modo()
        if nuevo_modo == "geocerca_poly":
            self.poly_toolbar.grid()
        else:
            self.poly_toolbar.grid_remove()

    def _invertir_lado_peligro(self):
        if self.config.get("zone_mode", "geocerca_poly") == "geocerca_poly":
            self._alternar_inversion_zona(self.selected_zone_idx)
            return
        self.config["invert_side"] = not self.config["invert_side"]
        self._guardar_config()
        estado_inv = "Invertido" if self.config["invert_side"] else "Estándar"
        self._registrar_evento("INFO", "Línea Invertida", f"Lado de peligro en línea cambiado a: {estado_inv}")

    def _refrescar_botones_modo(self):
        m = self.config.get("zone_mode", "geocerca_poly")
        if m == "geocerca_rect":
            m = "geocerca_poly"
            self.config["zone_mode"] = "geocerca_poly"
        self.btn_mode_poly.configure(
            bg=self.C_BLUE if m == "geocerca_poly" else "#1E293B",
            fg="white" if m == "geocerca_poly" else self.C_MUTED
        )
        self.btn_mode_vline.configure(
            bg=self.C_BLUE if m == "linea_vertical" else "#1E293B",
            fg="white" if m == "linea_vertical" else self.C_MUTED
        )
        self.btn_mode_hline.configure(
            bg=self.C_BLUE if m == "linea_horizontal" else "#1E293B",
            fg="white" if m == "linea_horizontal" else self.C_MUTED
        )
        if m == "geocerca_poly":
            self.poly_toolbar.grid()
        else:
            self.poly_toolbar.grid_remove()

    def _conectar_desde_barra_rapida(self):
        raw_ip = self.ent_quick_ip.get().strip()
        if not raw_ip:
            return
        url_norm = self._resolver_url_con_guardadas(raw_ip)
        self._activar_camara_como_principal(url_norm)

    def _guardar_calibracion_ui(self):
        raw_url = self.ent_cam_url.get().strip()
        url_norm = self._resolver_url_con_guardadas(raw_url)
        old_url = self.config["camera_url"]
        old_w = int(self.config.get("max_width", 640))
        new_w = int(self.var_max_width.get())

        # Extraer puerto COM seleccionado (ej. "COM9 — USB-SERIAL CH340" -> "COM9")
        raw_port = self.combo_esp32_port.get().strip() if hasattr(self, "combo_esp32_port") else "AUTO"
        if raw_port.upper().startswith("AUTO") or not raw_port:
            port_clean = "AUTO"
        else:
            port_clean = raw_port.split("—")[0].split()[0].strip().upper()

        esp_mode = self.var_esp32_mode.get() if hasattr(self, "var_esp32_mode") else "ambos"
        esp_ip = self.ent_esp32_ip.get().strip() if hasattr(self, "ent_esp32_ip") else ""
        esp_low = bool(self.var_esp32_low.get()) if hasattr(self, "var_esp32_low") else True

        self.config["camera_url"] = url_norm
        self.config["max_width"] = new_w
        self.config["detect_point"] = self.var_detect_pt.get()
        self.config["esp32_mode"] = esp_mode
        self.config["esp32_port"] = port_clean
        self.config["esp32_wifi_ip"] = esp_ip
        self.config["esp32_active_low"] = esp_low
        self.config["plc_ip"] = self.ent_plc_ip.get().strip()
        self.config["plc_port"] = self.ent_plc_port.get().strip()
        self.config["cell_name"] = self.ent_cell_name.get().strip()
        self.config["confidence"] = float(self.scale_conf.get())
        self.config["require_epp"] = bool(self.var_epp.get())
        self.config["sound_alarm"] = bool(self.var_sound.get())

        self._guardar_config()

        self.esp32.actualizar_config(
            mode=esp_mode,
            serial_port=port_clean,
            baudrate=115200,
            wifi_ip=esp_ip,
            active_low=esp_low
        )

        self.lbl_cell_title.configure(text=self.config["cell_name"])
        self.ent_quick_ip.delete(0, tk.END)
        self.ent_quick_ip.insert(0, url_norm)
        self.ent_cam_url.delete(0, tk.END)
        self.ent_cam_url.insert(0, url_norm)

        if url_norm != old_url or new_w != old_w:
            user, pwd = self._buscar_credenciales_guardadas(url_norm)
            self.cam = self.cam_pool.reconectar(url_norm, max_width=new_w, username=user, password=pwd)
            self._limpiar_pool_inactivas()
            self._registrar_evento("INFO", "Cámara Actualizada", f"Fuente: {url_norm} ({new_w}p)")

        messagebox.showinfo("Configuración Guardada", "Los parámetros de Cámara y ESP32 (4 Relés) se guardaron correctamente.")

    # -----------------------------------------------------------------
    # CONTROLES DE NAVEGACIÓN, SISTEMA ARMADO, PARO Y BITÁCORA
    # -----------------------------------------------------------------
    def _cambiar_vista(self, nombre_vista):
        self.vista_actual = nombre_vista
        for k, btn in self.nav_buttons.items():
            if k == nombre_vista:
                btn.configure(bg=self.C_BLUE, fg="white")
            else:
                btn.configure(bg=self.C_BG_SIDEBAR, fg=self.C_MUTED)

        self.view_monitoreo.grid_forget()
        self.view_multicam.grid_forget()
        self.view_camaras.grid_forget()
        self.view_bitacora.grid_forget()
        self.view_calibracion.grid_forget()

        if nombre_vista == "monitoreo":
            self.view_monitoreo.grid(row=0, column=0, sticky="nsew")
        elif nombre_vista == "multicam":
            self.view_multicam.grid(row=0, column=0, sticky="nsew")
            self._sincronizar_streams_quad()
        elif nombre_vista == "camaras":
            self.view_camaras.grid(row=0, column=0, sticky="nsew")
        elif nombre_vista == "bitacora":
            self.view_bitacora.grid(row=0, column=0, sticky="nsew")
        elif nombre_vista == "calibracion":
            self._refrescar_puertos_com_ui()
            self.view_calibracion.grid(row=0, column=0, sticky="nsew")

        self._limpiar_pool_inactivas()

    def _toggle_sistema_armado(self):
        self.sistema_armado = not self.sistema_armado
        if self.sistema_armado:
            self.btn_armado.configure(text="●  SISTEMA ARMADO", bg="#064E3B", fg="#34D399")
            self._registrar_evento("INFO", "Sistema Armado", "Monitoreo de geocercas y relés ESP32 activado.")
        else:
            self.btn_armado.configure(text="○  SISTEMA EN BYPASS", bg="#374151", fg="#FBBF24")
            self.esp32.establecer_estado_seguridad("BYPASS")
            self._registrar_evento("WARN", "Sistema Desarmado", "Monitoreo en modo Bypass.")

    def _toggle_paro_emergencia(self):
        self.paro_emergencia = not self.paro_emergencia
        if self.paro_emergencia:
            self.btn_paro.configure(text="🔓 RESTABLECER PARO", bg=self.C_AMBER, fg="#090D16")
            self.esp32.establecer_estado_seguridad("CORTE")
            self._registrar_evento("CRITICO", "Paro de Emergencia Manual", "Operador activó Corte de Corriente (Relé 2) + Alarma (Relé 1).")
            self._sonar_alarma()
        else:
            self.btn_paro.configure(text="🛑 PARO DE EMERGENCIA", bg=self.C_RED, fg="white")
            self._registrar_evento("INFO", "Paro Restablecido", "Enclavamiento liberado manualmente.")

    def _sonar_alarma(self):
        if not self.config.get("sound_alarm", True) or not HAS_SOUND:
            return
        ahora = time.time()
        if ahora - self.ultimo_beep_ts > 1.2:
            self.ultimo_beep_ts = ahora
            threading.Thread(target=lambda: winsound.Beep(1100, 180), daemon=True).start()

    def _registrar_evento(self, nivel, titulo, detalle):
        ahora = datetime.datetime.now()
        fecha_str = ahora.strftime("%Y-%m-%d")
        hora_str = ahora.strftime("%H:%M:%S")

        ev = {"fecha": fecha_str, "hora": hora_str, "nivel": nivel, "titulo": titulo, "detalle": detalle}
        self.bitacora_eventos.insert(0, ev)

        if hasattr(self, "tree_log"):
            self.tree_log.insert("", 0, values=(fecha_str, hora_str, nivel, titulo, detalle))

        self.alertas_recientes.insert(0, ev)
        self.alertas_recientes = self.alertas_recientes[:5]
        if hasattr(self, "alerts_list_frame"):
            self._renderizar_alertas_recientes()

    def _renderizar_alertas_recientes(self):
        for w in self.alerts_list_frame.winfo_children():
            w.destroy()

        if not self.alertas_recientes:
            tk.Label(
                self.alerts_list_frame, text="Sin alertas registradas",
                font=("Segoe UI", 9), bg=self.C_BG_CARD, fg=self.C_MUTED
            ).pack(pady=20)
            return

        for ev in self.alertas_recientes:
            es_crit = ev["nivel"] == "CRITICO"
            es_warn = ev["nivel"] == "WARN"
            borde = self.C_RED if es_crit else (self.C_AMBER if es_warn else "#2563EB")
            icono = "🚨" if es_crit else ("⚠️" if es_warn else "ℹ️")

            item = tk.Frame(self.alerts_list_frame, bg="#1E293B", padx=10, pady=7)
            item.pack(fill="x", pady=4)

            bar = tk.Frame(item, bg=borde, width=4)
            bar.pack(side="left", fill="y", padx=(0, 8))

            txt_f = tk.Frame(item, bg="#1E293B")
            txt_f.pack(side="left", fill="both", expand=True)

            top_r = tk.Frame(txt_f, bg="#1E293B")
            top_r.pack(fill="x")
            tk.Label(top_r, text=f"{icono} {ev['titulo']}", font=("Segoe UI", 9, "bold"), bg="#1E293B", fg=self.C_TEXT).pack(side="left")
            tk.Label(top_r, text=ev["hora"], font=("Consolas", 8), bg="#1E293B", fg=self.C_MUTED).pack(side="right")

            tk.Label(
                txt_f, text=ev["detalle"], font=("Segoe UI", 8),
                bg="#1E293B", fg=self.C_MUTED, wraplength=260, justify="left"
            ).pack(anchor="w", pady=(2, 0))

    def _limpiar_alertas_recientes(self):
        self.alertas_recientes.clear()
        self._renderizar_alertas_recientes()

    def _limpiar_bitacora(self):
        self.bitacora_eventos.clear()
        for item in self.tree_log.get_children():
            self.tree_log.delete(item)

    def _exportar_bitacora_csv(self):
        if not self.bitacora_eventos:
            messagebox.showinfo("Bitácora vacía", "No hay eventos para exportar.")
            return
        ruta = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Archivo CSV", "*.csv")],
            initialfile=f"bitacora_safevision_{datetime.date.today()}.csv"
        )
        if not ruta:
            return
        with open(ruta, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["fecha", "hora", "nivel", "titulo", "detalle"])
            writer.writeheader()
            writer.writerows(self.bitacora_eventos)
        messagebox.showinfo("Exportación Exitosa", f"Bitácora guardada en:\n{ruta}")

    # -----------------------------------------------------------------
    # GEOMETRÍA Y EVALUACIÓN POR ÁREA: ALARMA (RELÉ 1) vs CORTE (RELÉ 2)
    # -----------------------------------------------------------------
    def _evaluar_punto_en_zonas(self, px, py, w, h):
        """
        Devuelve (nivel_accion, nombre_zona) donde nivel_accion es:
          - "corte":  Persona en Área 2 (Cortar Corriente -> Relé 2 + Relé 1)
          - "alarma": Persona en Área 1 (Encender Alarma -> Relé 1)
          - "normal": Persona fuera de zonas de riesgo o dentro de Zona Segura
        """
        nx = px / float(w)
        ny = py / float(h)
        modo = self.config.get("zone_mode", "geocerca_poly")
        inv = self.config.get("invert_side", False)

        if modo == "linea_vertical":
            lx = self.config["line_x"]
            adentro = (nx < lx) if inv else (nx > lx)
            return ("corte" if adentro else "normal"), "Línea Vertical"

        if modo == "linea_horizontal":
            ly = self.config["line_y"]
            adentro = (ny < ly) if inv else (ny > ly)
            return ("corte" if adentro else "normal"), "Línea Horizontal"

        geofences = self.config.get("geofences", [])
        adentro_corte = False
        nombre_corte = ""
        adentro_alarma = False
        nombre_alarma = ""
        adentro_segura = False
        nombre_segura = ""
        total_riesgo = 0
        total_segura = 0

        for gz in geofences:
            pts = gz.get("points", [])
            if len(pts) < 3:
                continue
            acc = self._obtener_accion_zona(gz)
            if acc == "seguro":
                total_segura += 1
            else:
                total_riesgo += 1

            poly_px = np.array([[p[0] * w, p[1] * h] for p in pts], dtype=np.float32)
            if cv2.pointPolygonTest(poly_px, (float(px), float(py)), False) >= 0:
                if acc == "seguro":
                    adentro_segura = True
                    nombre_segura = gz.get("name", "Zona Segura")
                elif acc == "corte":
                    adentro_corte = True
                    nombre_corte = gz.get("name", "Zona Corte")
                elif acc == "alarma":
                    adentro_alarma = True
                    nombre_alarma = gz.get("name", "Zona Alarma")

        # Si está dentro de una isla explícitamente configurada como Zona Segura
        if adentro_segura:
            return "normal", f"{nombre_segura} (Seguro)"

        # Prioridad 1: Área de CORTE DE CORRIENTE (Relé 2 + Relé 1)
        if adentro_corte:
            return "corte", f"{nombre_corte} (Corte Corriente)"

        # Prioridad 2: Área de ENCENDER ALARMA (Relé 1)
        if adentro_alarma:
            return "alarma", f"{nombre_alarma} (Alarma)"

        # Si solo existen zonas SEGURAS y la persona sale fuera de ellas -> Corte de Corriente
        if total_segura > 0 and total_riesgo == 0:
            return "corte", "Fuera de Zona Segura"

        return "normal", "Zona Despejada"

    def _dibujar_zonas_y_handles(self, frame, estado_global):
        h, w = frame.shape[:2]
        modo = self.config.get("zone_mode", "geocerca_poly")
        inv = self.config.get("invert_side", False)
        hay_intrusion = estado_global in ("ALARMA", "CORTE")

        overlay = frame.copy()

        if modo == "linea_vertical":
            lx = int(self.config["line_x"] * w)
            color_line = (68, 68, 239) if hay_intrusion else (0, 200, 255)
            if not inv:
                cv2.rectangle(overlay, (lx, 0), (w, h), (40, 40, 220), -1)
            else:
                cv2.rectangle(overlay, (0, 0), (lx, h), (40, 40, 220), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.line(frame, (lx, 0), (lx, h), color_line, 3, cv2.LINE_AA)
            cv2.circle(frame, (lx, h // 2), 11, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "<-> ARRASTRAR LINEA", (max(10, lx - 85), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            return

        if modo == "linea_horizontal":
            ly = int(self.config["line_y"] * h)
            color_line = (68, 68, 239) if hay_intrusion else (0, 200, 255)
            if not inv:
                cv2.rectangle(overlay, (0, ly), (w, h), (40, 40, 220), -1)
            else:
                cv2.rectangle(overlay, (0, 0), (w, ly), (40, 40, 220), -1)
            cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            cv2.line(frame, (0, ly), (w, ly), color_line, 3, cv2.LINE_AA)
            cv2.circle(frame, (w // 2, ly), 11, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "^v ARRASTRAR LINEA", (20, max(25, ly - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            return

        geofences = self.config.get("geofences", [])
        for z_idx, gz in enumerate(geofences):
            pts = gz.get("points", [])
            if len(pts) < 3:
                continue
            acc = self._obtener_accion_zona(gz)
            pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in pts], dtype=np.int32)
            if acc == "seguro":
                fill_col = (110, 185, 16)   # Verde esmeralda (Zona Segura)
            elif acc == "alarma":
                fill_col = (11, 158, 245)   # Ámbar (Área 1: Encender Alarma - Relé 1)
            else:
                fill_col = (50, 50, 239)    # Rojo industrial (Área 2: Cortar Corriente - Relé 2)
            cv2.fillPoly(overlay, [pts_px], fill_col)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        for z_idx, gz in enumerate(geofences):
            pts = gz.get("points", [])
            if len(pts) < 3:
                continue
            es_activa = (z_idx == self.selected_zone_idx)
            acc = self._obtener_accion_zona(gz)
            pts_px = np.array([[int(p[0] * w), int(p[1] * h)] for p in pts], dtype=np.int32)
            if acc == "seguro":
                base_col = (129, 185, 16)
                estado_lbl = "ZONA SEGURA"
                bg_tag = (6, 78, 59)
            elif acc == "alarma":
                base_col = (11, 158, 245)
                estado_lbl = "1: ALARMA (RELE 1)"
                bg_tag = (18, 65, 120)
            else:
                base_col = (68, 68, 239)
                estado_lbl = "2: CORTE CORRIENTE (RELE 2)"
                bg_tag = (20, 20, 110)

            grosor = 3 if es_activa else 2
            cv2.polylines(frame, [pts_px], isClosed=True, color=base_col, thickness=grosor, lineType=cv2.LINE_AA)

            cx = int(np.mean(pts_px[:, 0]))
            cy = int(np.mean(pts_px[:, 1]))
            tag = f"{gz['name']} · {estado_lbl}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 6, cy - th - 6), (cx + tw // 2 + 6, cy + 6), bg_tag, -1)
            cv2.rectangle(frame, (cx - tw // 2 - 6, cy - th - 6), (cx + tw // 2 + 6, cy + 6), base_col, 1)
            cv2.putText(frame, tag, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            n = len(pts_px)
            if es_activa:
                for i in range(n):
                    mx = int((pts_px[i][0] + pts_px[(i + 1) % n][0]) * 0.5)
                    my = int((pts_px[i][1] + pts_px[(i + 1) % n][1]) * 0.5)
                    is_edge_hover = (
                        self.hover_info is not None
                        and self.hover_info[0] == "edge"
                        and self.hover_info[1] == z_idx
                        and self.hover_info[2] == i
                    )
                    r_mid = 8 if is_edge_hover else 5
                    col_mid = (0, 255, 180) if is_edge_hover else (220, 220, 220)
                    cv2.circle(frame, (mx, my), r_mid, (15, 23, 42), -1, cv2.LINE_AA)
                    cv2.circle(frame, (mx, my), r_mid, col_mid, 1, cv2.LINE_AA)
                    cv2.line(frame, (mx - 3, my), (mx + 3, my), col_mid, 1)
                    cv2.line(frame, (mx, my - 3), (mx, my + 3), col_mid, 1)

            for pt_idx, (vx, vy) in enumerate(pts_px):
                is_v_hover = (
                    self.hover_info is not None
                    and self.hover_info[0] == "vertex"
                    and self.hover_info[1] == z_idx
                    and self.hover_info[2] == pt_idx
                )
                r_v = 10 if is_v_hover else (8 if es_activa else 6)
                v_col = (0, 255, 255) if es_activa else (200, 200, 200)
                cv2.circle(frame, (vx, vy), r_v, v_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (vx, vy), r_v + 2, (15, 23, 42), 2, cv2.LINE_AA)
                if es_activa:
                    cv2.putText(frame, str(pt_idx + 1), (vx + 9, vy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------
    # BUCLE PRINCIPAL DE VIDEO, IA Y DISPARO DE 4 RELÉS ESP32
    # -----------------------------------------------------------------
    def _actualizar_loop(self):
        # Si estamos en la vista de 4 Cámaras, actualizar la cuadrícula 2x2 de forma optimizada
        if self.vista_actual == "multicam":
            self._actualizar_vista_multicam()
            self.root.after(35, self._actualizar_loop)
            return

        frame_raw, frame_id = self.cam.read() if self.cam else (None, -1)

        if frame_raw is not None and self.cam and self.cam.connected:
            self.lbl_cam_status.configure(
                text=f"● EN VIVO · {self.cam.fps_cam:.0f} FPS",
                fg=self.C_GREEN
            )
            if frame_id != self._last_frame_id and self.sistema_armado:
                self._last_frame_id = frame_id
                self.yolo_engine.conf_min = max(0.25, self.config["confidence"] / 100.0)
                self.yolo_engine.enviar_frame(frame_raw, frame_id, source_key="main")

            frame = frame_raw.copy()
        else:
            st_msg = self.cam.status_msg if self.cam else "Reconectando..."
            self.lbl_cam_status.configure(text=f"○ {st_msg[:48]}", fg=self.C_AMBER)
            frame = np.zeros((480, 800, 3), dtype=np.uint8)
            frame[:] = (22, 14, 9)

            # Ocultar contraseña en URL mostrada en pantalla
            url_disp = str(self.config.get("camera_url", ""))
            if "://" in url_disp and "@" in url_disp:
                p_part, rest_part = url_disp.split("://", 1)
                c_part, h_part = rest_part.rsplit("@", 1)
                u_only = c_part.split(":", 1)[0]
                url_disp = f"{p_part}://{u_only}:****@{h_part}"

            if self.cam and getattr(self.cam, "is_locked_error", False):
                mins, secs = divmod(int(getattr(self.cam, "unlock_seconds", 0)), 60)
                cv2.putText(frame, "CAMARA HIKVISION EN BLOQUEO TEMPORAL", (115, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (68, 68, 239), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Tiempo restante de desbloqueo: {mins}m {secs:02d}s", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (11, 158, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, "SOLUCION INMEDIATA: Desconecta y vuelve a conectar el cable", (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (129, 185, 16), 2, cv2.LINE_AA)
                cv2.putText(frame, "de corriente de la camara. SafeVision conectara automaticamente.", (105, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Fuente: {url_disp}", (180, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
            elif self.cam and getattr(self.cam, "auth_rejected", False):
                cv2.putText(frame, "USUARIO O CONTRASENA RECHAZADOS (401)", (125, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (68, 68, 239), 2, cv2.LINE_AA)
                cv2.putText(frame, "Se pausaron los reintentos para evitar bloquear la camara.", (135, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 158, 245), 1, cv2.LINE_AA)
                cv2.putText(frame, "Ve a 'Gestor de Camaras', corrige la clave y pulsa 'Guardar y Conectar'.", (85, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (129, 185, 16), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Fuente: {url_disp}", (180, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "CONECTANDO CAMARA ETHERNET / IP...", (165, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (11, 158, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Estado: {st_msg[:65]}", (140, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Fuente: {url_disp}", (180, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(frame, "Usa 'Gestor de Camaras -> Escanear' o activa 'Modo Prueba' en Configuracion.", (85, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (16, 185, 129), 1, cv2.LINE_AA)

        h, w = frame.shape[:2]
        personas_totales = 0
        personas_en_alarma = 0
        personas_en_corte = 0
        zona_activada = ""
        usar_pies = (self.config.get("detect_point", "pies") == "pies")

        dets, infer_ms = self.yolo_engine.obtener_detecciones("main") if (self.sistema_armado and frame_raw is not None) else ([], 0.0)
        conf_umbral = self.config["confidence"] / 100.0

        for (x1, y1, x2, y2, conf) in dets:
            if conf < conf_umbral:
                continue
            personas_totales += 1
            cx = (x1 + x2) // 2
            py_ref = max(y1, min(h - 1, y2 - 6)) if usar_pies else (y1 + y2) // 2

            nivel_acc, nombre_z = self._evaluar_punto_en_zonas(cx, py_ref, w, h)
            if nivel_acc == "corte":
                personas_en_corte += 1
                zona_activada = nombre_z
                color_box = (68, 68, 239)   # Rojo: Corte de Corriente
                tag = f"CORTE CORRIENTE [{nombre_z}] {conf * 100:.0f}%"
            elif nivel_acc == "alarma":
                personas_en_alarma += 1
                if not personas_en_corte:
                    zona_activada = nombre_z
                color_box = (11, 158, 245)  # Ámbar: Encender Alarma
                tag = f"ALARMA [{nombre_z}] {conf * 100:.0f}%"
            else:
                color_box = (129, 185, 16)  # Verde: Normal
                tag = f"Persona {conf * 100:.0f}%"

            rx1, ry1 = max(0, x1), max(0, y1)
            rx2, ry2 = min(w, x2), min(h, y2)
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                color_layer = np.full_like(roi, color_box)
                cv2.addWeighted(color_layer, 0.18, roi, 0.82, 0, roi)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, py_ref), 6, (0, 255, 255) if nivel_acc != "normal" else color_box, -1, cv2.LINE_AA)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            y_lbl = max(th + 8, y1)
            cv2.rectangle(frame, (x1, y_lbl - th - 8), (x1 + tw + 10, y_lbl), color_box, -1)
            cv2.putText(frame, tag, (x1 + 5, y_lbl - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Determinar estado de seguridad hacia la ESP32 (Prioridad: CORTE > ALARMA > NORMAL)
        if not self.sistema_armado:
            estado_esp32 = "BYPASS"
        elif self.paro_emergencia or personas_en_corte > 0:
            estado_esp32 = "CORTE"
        elif personas_en_alarma > 0:
            estado_esp32 = "ALARMA"
        else:
            estado_esp32 = "NORMAL"

        self.esp32.establecer_estado_seguridad(estado_esp32)

        self._dibujar_zonas_y_handles(frame, estado_esp32)

        self._actualizar_tarjetas_estado(
            personas_totales, personas_en_alarma, personas_en_corte, estado_esp32, infer_ms, zona_activada
        )

        if self.vista_actual == "monitoreo":
            cw, ch = self.canvas_w, self.canvas_h
            scale = min(cw / float(w), ch / float(h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            self.vid_w, self.vid_h = nw, nh
            self.vid_x0 = (cw - nw) // 2
            self.vid_y0 = (ch - nh) // 2

            frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_canvas.img_tk = img_tk
            self.video_canvas.delete("all")
            self.video_canvas.create_image(self.vid_x0, self.vid_y0, anchor="nw", image=img_tk)

        self._fps_frames += 1
        t_now = time.time()
        if t_now - self._fps_t0 >= 0.5:
            self.fps_actual = self._fps_frames / (t_now - self._fps_t0)
            self._fps_frames = 0
            self._fps_t0 = t_now

        self.root.after(15, self._actualizar_loop)

    def _actualizar_tarjetas_estado(self, p_total, p_alarma, p_corte, estado_esp32, infer_ms, zona_activada):
        self.lbl_metric_fps.configure(text=f"{self.fps_actual:.0f} FPS (IA: {infer_ms:.0f} ms)")
        n_zonas = len(self.config.get("geofences", []))
        self.lbl_metric_zones.configure(
            text=f"{n_zonas} {'geocerca' if n_zonas == 1 else 'geocercas'}"
            if self.config["zone_mode"] == "geocerca_poly"
            else self.config["zone_mode"].replace("_", " ").title()
        )

        # Actualizar indicadores visuales de los 4 relés de la ESP32
        r_states = self.esp32.relay_states
        colores_encendido = {
            1: ("#78350F", "#FCD34D"),  # Relé 1: Alarma (Ámbar)
            2: ("#7F1D1D", "#FCA5A5"),  # Relé 2: Corte de Corriente (Rojo)
            3: ("#064E3B", "#6EE7B7"),  # Relé 3: Luz Verde OK (Verde)
            4: ("#1E3A8A", "#93C5FD"),  # Relé 4: Auxiliar (Azul)
        }
        if hasattr(self, "relay_pills"):
            for num_r, nom_r, btn_r in self.relay_pills:
                enc = bool(r_states[num_r - 1])
                if enc:
                    bg_r, fg_r = colores_encendido.get(num_r, ("#1E293B", "white"))
                    btn_r.configure(text=f"{nom_r}\n● ON", bg=bg_r, fg=fg_r)
                else:
                    btn_r.configure(text=f"{nom_r}\n○ OFF", bg="#1E293B", fg="#6B7280")

        # Actualizar conexión ESP32 en Sidebar y tarjeta
        conectado_esp = self.esp32.usb_connected or self.esp32.wifi_connected
        if hasattr(self, "lbl_esp32_conn_mini"):
            if self.esp32.usb_connected and self.esp32.wifi_connected:
                txt_mini = f"● USB ({self.esp32.active_com_port}) + Wi-Fi"
            elif self.esp32.usb_connected:
                txt_mini = f"● USB {self.esp32.active_com_port} Conectado"
            elif self.esp32.wifi_connected:
                txt_mini = f"● Wi-Fi {self.esp32.wifi_ip}"
            else:
                txt_mini = "⏳ Buscando ESP32..."
            self.lbl_esp32_conn_mini.configure(
                text=txt_mini,
                fg=self.C_GREEN if conectado_esp else self.C_AMBER
            )

        self.lbl_plc_dot.configure(fg=self.C_GREEN if conectado_esp else self.C_AMBER)
        self.lbl_plc_title.configure(
            text=f"ESP32 ({self.esp32.active_com_port or 'USB/Wi-Fi'})" if conectado_esp else "ESP32 (Esperando conexión)"
        )
        self.lbl_plc_ip.configure(
            text=f"R1:{'ON' if r_states[0] else 'OFF'} | R2:{'ON' if r_states[1] else 'OFF'} | R3:{'ON' if r_states[2] else 'OFF'} | R4:{'ON' if r_states[3] else 'OFF'}"
        )

        if p_corte > 0:
            self.lbl_metric_persons.configure(text=f"{p_total} ({p_corte} EN CORTE CORRIENTE)", fg=self.C_RED)
        elif p_alarma > 0:
            self.lbl_metric_persons.configure(text=f"{p_total} ({p_alarma} EN ZONA ALARMA)", fg=self.C_AMBER)
        else:
            self.lbl_metric_persons.configure(text=f"{p_total} (0 en peligro)", fg=self.C_GREEN)

        if estado_esp32 == "CORTE":
            self.status_box.configure(bg="#3B0A0A", highlightbackground=self.C_RED)
            self.lbl_status_icon.configure(text="⚡", bg=self.C_RED)
            self.lbl_status_main.configure(text="CORRIENTE CORTADA", bg="#3B0A0A", fg="#F87171")
            motivo = "Paro Manual -> Relé 2 (Corte) + Relé 1 (Alarma)" if self.paro_emergencia else f"{zona_activada} -> Relé 2 (Corte) + Relé 1 ON"
            self.lbl_status_sub.configure(text=motivo, bg="#3B0A0A", fg="#FECACA")

            self._sonar_alarma()
            ahora = time.time()
            if p_corte > 0 and (ahora - self.ultimo_evento_ts > 3.5):
                self.ultimo_evento_ts = ahora
                self.total_intrusiones += 1
                self._registrar_evento(
                    "CRITICO",
                    f"Corte de Corriente — {zona_activada}",
                    f"{p_corte} persona(s) en área crítica. ESP32 activó Relé 2 (Corte de Corriente) y Relé 1 (Alarma)."
                )
        elif estado_esp32 == "ALARMA":
            self.status_box.configure(bg="#3B2506", highlightbackground=self.C_AMBER)
            self.lbl_status_icon.configure(text="🔔", bg=self.C_AMBER)
            self.lbl_status_main.configure(text="ALARMA ENCENDIDA", bg="#3B2506", fg="#FBBF24")
            self.lbl_status_sub.configure(
                text=f"{zona_activada} -> Relé 1 (Alarma) ON | Corriente OK",
                bg="#3B2506", fg="#FDE68A"
            )

            self._sonar_alarma()
            ahora = time.time()
            if p_alarma > 0 and (ahora - self.ultimo_evento_ts > 3.5):
                self.ultimo_evento_ts = ahora
                self.total_intrusiones += 1
                self._registrar_evento(
                    "WARN",
                    f"Alarma Activada — {zona_activada}",
                    f"{p_alarma} persona(s) en área de advertencia. ESP32 encendió Relé 1 (Alarma)."
                )
        elif not self.sistema_armado:
            self.status_box.configure(bg="#1F2937", highlightbackground="#4B5563")
            self.lbl_status_icon.configure(text="!", bg="#4B5563")
            self.lbl_status_main.configure(text="SISTEMA EN BYPASS", bg="#1F2937", fg="#D1D5DB")
            self.lbl_status_sub.configure(text="Monitoreo IA en pausa por operador", bg="#1F2937", fg="#9CA3AF")
        else:
            self.status_box.configure(bg="#062E22", highlightbackground="#059669")
            self.lbl_status_icon.configure(text="✓", bg="#059669")
            self.lbl_status_main.configure(text="MÁQUINA OPERANDO", bg="#062E22", fg="#34D399")
            self.lbl_status_sub.configure(text="Zonas Despejadas -> Relé 3 (Verde) ON | Corriente Activa", bg="#062E22", fg="#A7F3D0")


# Funciones de compatibilidad para scripts externos (ej. line_crossing.py)
def cargar_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"camera_url": "rtsp://admin:JorgePuta@192.168.1.64:554/Streaming/Channels/102", "confidence": 30.0, "sound_alarm": True, "line_x": 0.5}


def guardar_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def normalizar_entrada_camara(texto):
    return str(SmartCamaraIP.normalizar_url(texto))


def emitir_pitido():
    if HAS_SOUND:
        threading.Thread(target=lambda: winsound.Beep(1100, 180), daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = SafeVisionApp(root)
    root.mainloop()
