import ctypes
from ctypes import wintypes
import json
import os
import re
import subprocess
import threading
import time

try:
    import serial
    import serial.tools.list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False


# =====================================================================
# COMUNICACIÓN SERIAL USB NATIVA EN WINDOWS (CH340 / CP210x / ESP32)
# =====================================================================
class _DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fFlags", wintypes.DWORD),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


class Win32SerialPort:
    """
    Comunicación Serial UART nativa en Windows usando kernel32.dll (ctypes).
    Especialmente diseñada para placas ESP32 con chip CH340 / CP210x:
    - Desactiva DTR y RTS (CLRDTR + CLRRTS) al abrir para que el circuito de
      auto-reset de la ESP32 nunca deje el pin EN en bajo (Reset bloqueado).
    - Espera el arranque inicial de la ESP32 tras abrir el puerto.
    """
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    CLRRTS = 4
    CLRDTR = 6

    def __init__(self, port, baudrate=115200, timeout=0.05):
        self.port = str(port).strip().upper()
        self.baudrate = int(baudrate)
        self.timeout_ms = max(25, int(timeout * 1000))
        self.handle = None
        self.is_open = False
        self._open()

    def _open(self):
        if os.name != "nt":
            raise RuntimeError("Win32SerialPort solo está disponible en Windows.")
        k32 = ctypes.windll.kernel32

        k32.CreateFileW.restype = ctypes.c_void_p
        k32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p
        ]
        k32.GetCommState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DCB)]
        k32.GetCommState.restype = wintypes.BOOL
        k32.SetCommState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DCB)]
        k32.SetCommState.restype = wintypes.BOOL
        k32.EscapeCommFunction.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        k32.EscapeCommFunction.restype = wintypes.BOOL
        k32.SetCommTimeouts.argtypes = [ctypes.c_void_p, ctypes.POINTER(_COMMTIMEOUTS)]
        k32.SetCommTimeouts.restype = wintypes.BOOL
        k32.PurgeComm.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        k32.PurgeComm.restype = wintypes.BOOL
        k32.WriteFile.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
        ]
        k32.WriteFile.restype = wintypes.BOOL
        k32.ReadFile.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
        ]
        k32.ReadFile.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.CloseHandle.restype = wintypes.BOOL

        device_path = f"\\\\.\\{self.port}"
        h = k32.CreateFileW(
            device_path,
            self.GENERIC_READ | self.GENERIC_WRITE,
            0, None, self.OPEN_EXISTING, 0, None
        )
        if h is None or h == self.INVALID_HANDLE_VALUE:
            raise OSError(f"No se pudo abrir el puerto {self.port}")

        self.handle = ctypes.c_void_p(h)

        dcb = _DCB()
        dcb.DCBlength = ctypes.sizeof(_DCB)
        if not k32.GetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise OSError(f"Error en GetCommState para {self.port}")

        dcb.BaudRate = self.baudrate
        dcb.ByteSize = 8
        dcb.Parity = 0    # NOPARITY
        dcb.StopBits = 0  # ONESTOPBIT
        dcb.fFlags = 0x0001  # fBinary = 1, DTR_CONTROL_DISABLE, RTS_CONTROL_DISABLE
        if not k32.SetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise OSError(f"Error en SetCommState para {self.port}")

        # Liberar explícitamente DTR y RTS para que el chip CH340/CP210x deje correr la ESP32
        k32.EscapeCommFunction(self.handle, self.CLRDTR)
        k32.EscapeCommFunction(self.handle, self.CLRRTS)

        timeouts = _COMMTIMEOUTS()
        timeouts.ReadIntervalTimeout = 10
        timeouts.ReadTotalTimeoutMultiplier = 0
        timeouts.ReadTotalTimeoutConstant = self.timeout_ms
        timeouts.WriteTotalTimeoutMultiplier = 0
        timeouts.WriteTotalTimeoutConstant = 100
        k32.SetCommTimeouts(self.handle, ctypes.byref(timeouts))

        # Esperar 0.7s al abrir por primera vez por si el CH340 reinició la ESP32 al conectar
        time.sleep(0.70)
        k32.PurgeComm(self.handle, 0x000C)  # PURGE_RXCLEAR | PURGE_TXCLEAR
        self.is_open = True

    def write(self, data_bytes):
        if not self.is_open or not self.handle:
            raise OSError("Puerto cerrado")
        k32 = ctypes.windll.kernel32
        written = wintypes.DWORD(0)
        ok = k32.WriteFile(
            self.handle,
            data_bytes,
            len(data_bytes),
            ctypes.byref(written),
            None
        )
        if not ok or written.value == 0:
            raise OSError("Fallo al escribir en puerto COM")
        return written.value

    def readline(self):
        if not self.is_open or not self.handle:
            return b""
        k32 = ctypes.windll.kernel32
        buf = ctypes.create_string_buffer(256)
        read_bytes = wintypes.DWORD(0)
        ok = k32.ReadFile(self.handle, buf, 255, ctypes.byref(read_bytes), None)
        if not ok or read_bytes.value == 0:
            return b""
        raw = buf.raw[:read_bytes.value]
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return lines[-1] if lines else raw

    def close(self):
        if self.handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self.handle)
            except Exception:
                pass
        self.handle = None
        self.is_open = False


# =====================================================================
# CONTROLADOR ESP32 (MODO 100% USB SERIAL DIRECTO Y CONFIABLE)
# =====================================================================
class ControladorESP32Reles:
    """
    Controla la placa de 4 relés conectada a la ESP32 exclusivamente por cable USB
    (Puerto Serial COM a 115200 baudios), garantizando cero retrasos y reconexión
    automática en cualquier puerto USB donde se conecte la ESP32.

    Asignación de los 4 Relés:
      - Relé 1 (GPIO 26): 🔔 Alarma / Sirena (Área 1: Encender Alarma)
      - Relé 2 (GPIO 27): ⚡ Corte de Corriente (Área 2: Cortar Corriente)
      - Relé 3 (GPIO 14): 🟢 Luz Verde / Sistema Seguro (Zonas Despejadas)
      - Relé 4 (GPIO 25): 🔧 Auxiliar / Baliza
    """
    _wmi_cache = {}
    _wmi_cache_ts = 0.0

    def __init__(self, mode="usb", serial_port="AUTO", baudrate=115200, wifi_ip="", active_low=True):
        self.mode = "usb"  # Forzar comunicación 100% por USB
        self.serial_port = str(serial_port).strip().upper() or "AUTO"
        self.baudrate = int(baudrate)
        self.wifi_ip = ""
        self.active_low = bool(active_low)

        self.stopped = False
        self.usb_connected = False
        self.wifi_connected = False
        self.active_com_port = ""
        self.status_text = "Buscando ESP32 por cable USB..."

        # Estado actual de los 4 relés: [R1_Alarma, R2_Corte, R3_Verde, R4_Aux]
        self.relay_states = [False, False, True, False]
        self.current_state_name = "NORMAL"

        self._desired_state = "NORMAL"
        self._last_sent_state = None
        self._last_send_ts = 0.0
        self._manual_override_until = 0.0
        self._pending_raw_cmd = None

        self._ser_obj = None
        self._lock = threading.Lock()
        self._wake_event = threading.Event()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    @classmethod
    def _obtener_mapa_nombres_wmi(cls):
        """Obtiene nombres amigables de los puertos COM desde WMI con caché de 3 segundos."""
        now = time.time()
        if now - cls._wmi_cache_ts < 3.0 and cls._wmi_cache:
            return cls._wmi_cache
        mapa = {}
        if os.name == "nt":
            try:
                cmd = [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_PnPEntity -Filter \"Name LIKE '%(COM%'\" | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress"
                ]
                out = subprocess.check_output(
                    cmd, timeout=2.0, creationflags=subprocess.CREATE_NO_WINDOW
                ).decode("utf-8", errors="ignore").strip()
                if out:
                    data = json.loads(out)
                    if isinstance(data, dict):
                        data = [data]
                    for row in data:
                        nm = str(row.get("Name", "")).strip()
                        pnp = str(row.get("PNPDeviceID", "")).strip()
                        m = re.search(r"\((COM\d+)\)", nm, re.IGNORECASE)
                        if m:
                            mapa[m.group(1).upper()] = (nm, pnp)
            except Exception:
                pass
        cls._wmi_cache = mapa
        cls._wmi_cache_ts = now
        return mapa

    @classmethod
    def escanear_puertos_com(cls):
        """
        Devuelve ÚNICAMENTE los puertos COM que están FÍSICAMENTE CONECTADOS AHORA MISMO
        según HARDWARE\\DEVICEMAP\\SERIALCOMM en Windows, priorizando chips USB (CH340/CP210x)
        y descartando puertos Bluetooth fantasma.
        """
        encontrados = []
        vistos = set()

        # 1) Leer del Registro de Windows los puertos COM realmente activos en este instante
        activos_reg = {}
        if os.name == "nt":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
                i = 0
                while True:
                    try:
                        dev_name, com_val, _ = winreg.EnumValue(key, i)
                        i += 1
                        p_name = str(com_val).strip().upper()
                        if p_name:
                            activos_reg[p_name] = str(dev_name)
                    except OSError:
                        break
                winreg.CloseKey(key)
            except Exception:
                pass

        # 2) Enriquecer con pyserial si está disponible
        if HAS_PYSERIAL:
            try:
                for p in serial.tools.list_ports.comports():
                    port_name = str(p.device).strip().upper()
                    desc = str(p.description or port_name)
                    hwid = str(getattr(p, "hwid", "") or "")
                    es_bt = ("bthenum" in hwid.lower()) or ("bluetooth" in desc.lower())
                    es_esp = (not es_bt) and any(
                        k in (desc + " " + hwid).lower()
                        for k in ("ch340", "cp210", "ch910", "ftdi", "usb", "uart", "esp32", "vid_1a86", "vid_10c4")
                    )
                    encontrados.append({
                        "port": port_name,
                        "desc": f"{port_name} — {desc}",
                        "es_esp32": es_esp,
                        "es_bluetooth": es_bt
                    })
                    vistos.add(port_name)
            except Exception:
                pass

        # 3) Procesar los puertos activos del Registro de Windows
        mapa_wmi = cls._obtener_mapa_nombres_wmi() if activos_reg else {}
        for port_name, dev_kernel in activos_reg.items():
            if port_name in vistos:
                continue
            wmi_name, pnp_id = mapa_wmi.get(port_name, ("", ""))
            texto_total = f"{dev_kernel} {wmi_name} {pnp_id}".lower()
            es_bt = ("bthenum" in texto_total) or ("bluetooth" in texto_total) or ("bthmodem" in texto_total)
            es_usb = (not es_bt) and any(
                k in texto_total
                for k in ("ch340", "cp210", "vcp", "usb", "slab", "wch", "serial", "vid_1a86", "vid_10c4")
            )
            etiqueta = wmi_name if wmi_name else ("USB-SERIAL ESP32" if es_usb else dev_kernel.split("\\")[-1])
            encontrados.append({
                "port": port_name,
                "desc": f"{port_name} — {etiqueta}",
                "es_esp32": es_usb,
                "es_bluetooth": es_bt
            })
            vistos.add(port_name)

        # Ordenar: primero puertos USB/ESP32 reales, al final Bluetooth
        encontrados.sort(key=lambda x: (x.get("es_bluetooth", False), not x.get("es_esp32", False), x["port"]))
        return encontrados

    def actualizar_config(self, mode="usb", serial_port="AUTO", baudrate=115200, wifi_ip="", active_low=True):
        with self._lock:
            nuevo_p = str(serial_port).strip().upper() or "AUTO"
            cambio_serial = (nuevo_p != self.serial_port) or (int(baudrate) != self.baudrate)
            self.mode = "usb"
            self.serial_port = nuevo_p
            self.baudrate = int(baudrate)
            self.active_low = bool(active_low)
            self._last_sent_state = None
            if cambio_serial:
                self._cerrar_serial()
            self._pending_raw_cmd = f"LOW:{1 if self.active_low else 0}"
        self._wake_event.set()

    def establecer_estado_seguridad(self, nuevo_estado):
        """
        Establece el estado general según las zonas de YOLO:
        - "NORMAL": Zonas despejadas -> R1=OFF, R2=OFF, R3=ON (Luz Verde), R4=OFF
        - "ALARMA": Persona en Zona 1 (Alarma) -> R1=ON (Sirena), R2=OFF, R3=OFF, R4=OFF
        - "CORTE":  Persona en Zona 2 (Corte de Corriente) o Paro -> R1=ON, R2=ON (Corta corriente), R3=OFF, R4=ON
        - "BYPASS": Sistema desarmado -> Todos en reposo
        """
        st = str(nuevo_estado).strip().upper()
        if st not in ("NORMAL", "ALARMA", "CORTE", "BYPASS"):
            st = "NORMAL"

        if time.time() >= self._manual_override_until:
            self.current_state_name = st
            if st == "NORMAL":
                self.relay_states = [False, False, True, False]
            elif st == "ALARMA":
                self.relay_states = [True, False, False, False]
            elif st == "CORTE":
                self.relay_states = [True, True, False, True]
            elif st == "BYPASS":
                self.relay_states = [False, False, False, False]

        with self._lock:
            if st != self._desired_state:
                self._desired_state = st
                self._wake_event.set()

    def probar_rele_manual(self, num_rele, encender=None, duracion_seg=4.0):
        idx = int(num_rele) - 1
        if not (0 <= idx <= 3):
            return
        if encender is None:
            nuevo_val = not self.relay_states[idx]
        else:
            nuevo_val = bool(encender)

        self.relay_states[idx] = nuevo_val
        self.current_state_name = "MANUAL"
        self._manual_override_until = time.time() + float(duracion_seg)

        with self._lock:
            self._pending_raw_cmd = f"R{idx + 1}:{1 if nuevo_val else 0}"
        self._wake_event.set()

    def _abrir_serial_si_necesario(self):
        if self._ser_obj is not None:
            return True

        puertos_activos = self.escanear_puertos_com()
        candidatos = []

        # 1) Si el usuario eligió un puerto específico y está conectado, probarlo primero
        if self.serial_port and self.serial_port != "AUTO":
            candidatos.append(self.serial_port)

        # 2) Agregar automáticamente cualquier puerto USB-Serial (CH340 / CP210x) activo
        #    para que si cambió de puerto USB en la laptop (ej. de COM11 a COM9 o COM8), conecte solo
        for item in puertos_activos:
            if not item.get("es_bluetooth", False) and item["port"] not in candidatos:
                candidatos.append(item["port"])

        for p_name in candidatos:
            try:
                if HAS_PYSERIAL:
                    ser = serial.Serial()
                    ser.port = p_name
                    ser.baudrate = self.baudrate
                    ser.timeout = 0.05
                    ser.write_timeout = 0.15
                    ser.dtr = False
                    ser.rts = False
                    ser.open()
                    time.sleep(0.70)
                    ser.reset_input_buffer()
                else:
                    ser = Win32SerialPort(p_name, baudrate=self.baudrate, timeout=0.05)

                self._ser_obj = ser
                self.usb_connected = True
                self.active_com_port = p_name
                # Sincronizar polaridad Active-LOW al conectar
                try:
                    cmd_low = f"LOW:{1 if self.active_low else 0}\n".encode("ascii")
                    self._ser_obj.write(cmd_low)
                except Exception:
                    pass
                return True
            except Exception:
                continue

        self.usb_connected = False
        self.active_com_port = ""
        return False

    def _cerrar_serial(self):
        if self._ser_obj is not None:
            try:
                self._ser_obj.close()
            except Exception:
                pass
        self._ser_obj = None
        self.usb_connected = False
        self.active_com_port = ""

    def _procesar_respuesta_json(self, texto):
        if not texto:
            return
        try:
            inicio = texto.find("{")
            fin = texto.rfind("}")
            if inicio != -1 and fin != -1 and fin > inicio:
                data = json.loads(texto[inicio:fin + 1])
                if data.get("ok"):
                    self.relay_states = [
                        bool(data.get("r1", 0)),
                        bool(data.get("r2", 0)),
                        bool(data.get("r3", 0)),
                        bool(data.get("r4", 0)),
                    ]
                    if "state" in data:
                        self.current_state_name = str(data["state"])
        except Exception:
            pass

    def _enviar_por_usb(self, comando_str):
        if not self._abrir_serial_si_necesario():
            return False
        try:
            payload = (comando_str.strip() + "\n").encode("ascii", errors="ignore")
            self._ser_obj.write(payload)
            resp = self._ser_obj.readline()
            if resp:
                self._procesar_respuesta_json(resp.decode("utf-8", errors="ignore"))
            self.usb_connected = True
            return True
        except Exception:
            self._cerrar_serial()
            return False

    def _actualizar_texto_estado(self):
        if self.usb_connected and self.active_com_port:
            self.status_text = f"🟢 ESP32 Conectada por USB ({self.active_com_port})"
        else:
            self.status_text = "🔌 Conecta la ESP32 por cable USB (Buscando puerto COM...)"

    def _worker_loop(self):
        while not self.stopped:
            self._wake_event.wait(timeout=0.25)
            self._wake_event.clear()
            if self.stopped:
                break

            ahora = time.time()
            with self._lock:
                raw_cmd = self._pending_raw_cmd
                self._pending_raw_cmd = None
                target_st = self._desired_state

            # 1) Si hay un comando manual directo (ej. prueba de un relé individual R1:1)
            if raw_cmd:
                self._enviar_por_usb(raw_cmd)
                self._actualizar_texto_estado()
                continue

            # 2) Si estamos en pausa por prueba manual de relé (4 seg), mantener el relé probado
            if ahora < self._manual_override_until:
                continue

            # 3) Enviar cambio de estado al instante, y refrescar cada 0.25s por seguridad continua
            if target_st != self._last_sent_state or (ahora - self._last_send_ts >= 0.25):
                cmd_estado = f"STATE:{target_st}"
                ok_u = self._enviar_por_usb(cmd_estado)
                if ok_u:
                    self._last_sent_state = target_st
                self._last_send_ts = ahora
                self._actualizar_texto_estado()

    def stop(self):
        self.stopped = True
        self._wake_event.set()
        self._cerrar_serial()
