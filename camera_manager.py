import cv2
import numpy as np
import urllib.request
import urllib.error
import urllib.parse
import base64
import hashlib
import threading
import time
import socket
import subprocess
import os
import re
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurar FFmpeg en OpenCV para cámaras RTSP / Ethernet sobre TCP y silenciar warnings de escaneo USB
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
try:
    cv2.setLogLevel(0)
except Exception:
    pass

CAMERA_PROFILES = [
    ("Auto / IP Webcam (/video)", "http", "8080", "/video"),
    ("DroidCam Celular (/video)", "http", "4747", "/video"),
    ("Webcam USB / Integrada (Índice 0, 1...)", "usb", "", ""),
    ("Hikvision / HiLook RTSP Fluido (Ch 102)", "rtsp", "554", "/Streaming/Channels/102"),
    ("Hikvision / HiLook RTSP HD (Ch 101)", "rtsp", "554", "/Streaming/Channels/101"),
    ("Dahua / Amcrest RTSP Sub-Stream", "rtsp", "554", "/cam/realmonitor?channel=1&subtype=1"),
    ("TP-Link Tapo RTSP Fluido (/stream2)", "rtsp", "554", "/stream2"),
    ("TP-Link Tapo RTSP HD (/stream1)", "rtsp", "554", "/stream1"),
    ("Steren / Genérica ONVIF RTSP (/live/ch0)", "rtsp", "554", "/live/ch0"),
    ("Ezviz / Reolink RTSP (/h264Preview_01_sub)", "rtsp", "554", "/h264Preview_01_sub"),
    ("Motion / MJPEG HTTP (:8081)", "http", "8081", "/"),
]


def construir_url_camara(cam_dict):
    """
    Construye la URL final o índice USB a partir de los datos de registro de una cámara.
    Soporta cámaras USB, HTTP MJPEG y RTSP con o sin usuario/contraseña.
    """
    tipo = str(cam_dict.get("type", "http")).lower()
    host = str(cam_dict.get("host", "")).strip().replace(";", ":")
    port = str(cam_dict.get("port", "")).strip()
    user = str(cam_dict.get("username", "")).strip()
    pwd = str(cam_dict.get("password", "")).strip()
    path = str(cam_dict.get("stream_path", "")).strip()

    if not host:
        return "0"

    # Si es índice USB puro ("0", "1", "2"...) o tipo usb
    if host.isdigit() or tipo == "usb":
        digits = "".join(ch for ch in host if ch.isdigit())
        return digits if digits else "0"

    # Si el usuario ya pegó una URL completa en host
    if host.startswith(("http://", "https://", "rtsp://")):
        if user and "@" not in host:
            proto, resto = host.split("://", 1)
            auth_str = f"{user}:{pwd}@" if pwd else f"{user}@"
            return f"{proto}://{auth_str}{resto}"
        return host

    proto = "rtsp" if ("rtsp" in tipo or port in ("554", "8554", "8000")) else "http"
    if port == "8000":
        port = "554"

    auth_part = ""
    if user:
        auth_part = f"{user}:{pwd}@" if pwd else f"{user}@"

    # Si el host ya incluye :puerto, separarlo
    if ":" in host:
        h_parts = host.split(":", 1)
        host = h_parts[0]
        if not port:
            port = h_parts[1].split("/")[0]

    port_part = f":{port}" if port else ""
    if not path:
        path = "/video" if proto == "http" else "/Streaming/Channels/102"
    if not path.startswith("/"):
        path = "/" + path

    return f"{proto}://{auth_part}{host}{port_part}{path}"


class SmartCamaraIP:
    """
    Lector universal de cámaras con 0-delay:
    - Webcams USB (prueba CAP_DSHOW, CAP_MSMF y CAP_ANY para máxima compatibilidad)
    - Cámaras RTSP / Ethernet / Hikvision / ONVIF (con pre-verificación Digest Auth,
      auto-detección de canal 102/101, fallback ISAPI HTTP y protección anti-bloqueo)
    - Cámaras HTTP MJPEG (IP Webcam, DroidCam, etc. con soporte Basic/Digest Auth y drenado TCP)
    """
    def __init__(self, url_or_index, max_width=640, username="", password=""):
        self.raw_input = str(url_or_index).strip()
        self.max_width = int(max_width)
        self.username = str(username).strip()
        self.password = str(password).strip()

        self.frame = None
        self.stopped = False
        self.connected = False
        self.requires_auth_error = False
        self.auth_rejected = False
        self.is_locked_error = False
        self.unlock_seconds = 0
        self.status_msg = "Iniciando..."
        self.fps_cam = 0.0
        self.lock = threading.Lock()
        self._frame_id = 0

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    @staticmethod
    def normalizar_url(texto):
        t = str(texto).strip().replace(";", ":")
        if t.isdigit():
            return int(t)
        if not t.startswith(("http://", "https://", "rtsp://")):
            if ":554" in t or ":8554" in t or "/Streaming/" in t or "/cam/realmonitor" in t or "/live/" in t:
                t = "rtsp://" + t
            else:
                t = "http://" + t
        if t.startswith("http"):
            sin_proto = t.split("://", 1)[1]
            if "/" not in sin_proto or sin_proto.endswith("/"):
                t = t.rstrip("/") + "/video"
        return t

    def _extraer_credenciales_de_url(self, url):
        """Separa proto://user:pass@host:port/path devolviendo (clean_url, user, pwd)."""
        user, pwd = self.username, self.password
        clean_url = str(url)
        if "://" in clean_url:
            proto, rest = clean_url.split("://", 1)
            if "@" in rest:
                creds, hostpart = rest.rsplit("@", 1)
                if ":" in creds:
                    u, p = creds.split(":", 1)
                else:
                    u, p = creds, ""
                u = urllib.parse.unquote(u)
                p = urllib.parse.unquote(p)
                if not user:
                    user = u
                if not pwd and p:
                    pwd = p
                clean_url = f"{proto}://{hostpart}"
        return clean_url, user, pwd

    @staticmethod
    def _parsear_host_puerto_ruta(clean_url, default_port=554, default_path="/Streaming/Channels/102"):
        """Extrae (host, port_int, path) de una URL sin credenciales."""
        sin_proto = clean_url.split("://", 1)[1] if "://" in clean_url else clean_url
        if "/" in sin_proto:
            hostport, path = sin_proto.split("/", 1)
            path = "/" + path
        else:
            hostport, path = sin_proto, default_path
        if ":" in hostport:
            host, port_s = hostport.split(":", 1)
            try:
                port = int(port_s)
            except Exception:
                port = default_port
        else:
            host, port = hostport, default_port
        return host.strip(), port, path

    @staticmethod
    def _construir_rtsp_con_auth(host, port, path, user="", pwd=""):
        if not path.startswith("/"):
            path = "/" + path
        if user:
            u_enc = urllib.parse.quote(user, safe="")
            p_enc = urllib.parse.quote(pwd, safe="")
            auth = f"{u_enc}:{p_enc}@" if pwd else f"{u_enc}@"
        else:
            auth = ""
        return f"rtsp://{auth}{host}:{port}{path}"

    def _redimensionar_si_necesario(self, img):
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > self.max_width:
            scale = self.max_width / float(w)
            img = cv2.resize(img, (self.max_width, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        return img

    def _abrir_webcam_usb(self, idx):
        """Prueba múltiples backends de Windows para abrir webcams externas e integradas."""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for b in backends:
            if self.stopped:
                break
            try:
                cap = cv2.VideoCapture(idx, b)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    for _ in range(4):
                        ret, frm = cap.read()
                        if ret and frm is not None and frm.size > 0:
                            return cap, frm
                        time.sleep(0.03)
                cap.release()
            except Exception:
                pass
        return None, None

    @staticmethod
    def _consultar_bloqueo_hikvision(host, user="admin", pwd=""):
        """
        Comprueba si una cámara Hikvision/HiLook reporta <lockStatus>lock</lockStatus> y su <unlockTime>.
        Solo se invoca UNA vez tras recibir 401 en RTSP para informar con precisión al usuario.
        """
        try:
            url_cap = f"http://{host}/ISAPI/Security/sessionLogin/capabilities?username={urllib.parse.quote(user or 'admin')}"
            cap_bytes = urllib.request.urlopen(url_cap, timeout=1.5).read()
            root = ET.fromstring(cap_bytes.decode("utf-8", errors="ignore"))
            ns = {"ns": "http://www.hikvision.com/ver20/XMLSchema"}
            sid_el = root.find("ns:sessionID", ns)
            chal_el = root.find("ns:challenge", ns)
            iter_el = root.find("ns:iterations", ns)
            salt_el = root.find("ns:salt", ns)
            irrev_el = root.find("ns:isIrreversible", ns)

            if sid_el is None or chal_el is None or iter_el is None:
                return False, 0

            sid = sid_el.text or ""
            chal = chal_el.text or ""
            iters = int(iter_el.text or "100")
            salt = (salt_el.text or "") if salt_el is not None else ""
            irrev = (irrev_el.text == "true") if irrev_el is not None else False

            u_str = user or "admin"
            if irrev:
                p_hash = hashlib.sha256((u_str + salt + pwd).encode("utf-8")).hexdigest()
                p_hash = hashlib.sha256((p_hash + chal).encode("utf-8")).hexdigest()
                for _ in range(iters - 2):
                    p_hash = hashlib.sha256(p_hash.encode("utf-8")).hexdigest()
            else:
                p_hash = hashlib.sha256(pwd.encode("utf-8")).hexdigest() + chal
                for _ in range(iters):
                    p_hash = hashlib.sha256(p_hash.encode("utf-8")).hexdigest()

            xml_body = (
                f"<SessionLogin><userName>{u_str}</userName>"
                f"<password>{p_hash}</password>"
                f"<sessionID>{sid}</sessionID>"
                f"<isSessionIDValidLongTerm>false</isSessionIDValidLongTerm>"
                f"<sessionIDVersion>2</sessionIDVersion></SessionLogin>"
            ).encode("utf-8")

            req = urllib.request.Request(
                f"http://{host}/ISAPI/Security/sessionLogin?timeStamp={int(time.time()*1000)}",
                data=xml_body,
                method="POST",
                headers={"Content-Type": "application/xml"}
            )
            try:
                urllib.request.urlopen(req, timeout=1.8).read()
                return False, 0
            except urllib.error.HTTPError as he:
                err_xml = he.read().decode("utf-8", errors="ignore")
                m_lock = re.search(r"<lockStatus>\s*lock\s*</lockStatus>", err_xml, re.IGNORECASE)
                m_time = re.search(r"<unlockTime>\s*(\d+)\s*</unlockTime>", err_xml, re.IGNORECASE)
                if m_lock:
                    secs = int(m_time.group(1)) if m_time else 600
                    return True, secs
        except Exception:
            pass
        return False, 0

    @staticmethod
    def _probar_rtsp_describe(host, port, path, user="", pwd="", timeout=2.0):
        """
        Realiza un único apretón de manos RTSP DESCRIBE (con soporte Basic y Digest Auth)
        para verificar credenciales y ruta SIN ciclar FFmpeg ni bloquear cámaras Hikvision.
        Devuelve: (status_code, mensaje)
          - 200: Stream válido y autenticado
          - 401: No autorizado (requiere clave o clave incorrecta)
          - 404: Ruta de stream no existe (pero credenciales válidas)
          - 0:   Error de red / puerto cerrado
        """
        uri = f"rtsp://{host}:{port}{path}"
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                req1 = f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: SafeVision/4.0\r\nAccept: application/sdp\r\n\r\n"
                sock.sendall(req1.encode("utf-8"))
                resp1 = sock.recv(4096).decode("utf-8", errors="ignore")
                if not resp1:
                    return 0, "Sin respuesta RTSP"

                first_line = resp1.splitlines()[0] if resp1.splitlines() else ""
                if " 200 " in first_line:
                    return 200, "OK"
                if " 404 " in first_line:
                    return 404, "Ruta no encontrada"
                if " 401 " not in first_line and "WWW-Authenticate" not in resp1:
                    return 200, first_line

                # Pide autenticación (401)
                if not user or not pwd:
                    return 401, "Falta usuario y contraseña"

                auth_header = None
                m_realm = re.search(r'Digest\s+.*?realm="([^"]+)"', resp1, re.IGNORECASE)
                m_nonce = re.search(r'Digest\s+.*?nonce="([^"]+)"', resp1, re.IGNORECASE)
                if m_realm and m_nonce:
                    realm = m_realm.group(1)
                    nonce = m_nonce.group(1)
                    ha1 = hashlib.md5(f"{user}:{realm}:{pwd}".encode("utf-8")).hexdigest()
                    ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode("utf-8")).hexdigest()
                    response_hash = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()
                    auth_header = (
                        f'Digest username="{user}", realm="{realm}", '
                        f'nonce="{nonce}", uri="{uri}", response="{response_hash}"'
                    )
                else:
                    b64 = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
                    auth_header = f"Basic {b64}"

                req2 = (
                    f"DESCRIBE {uri} RTSP/1.0\r\n"
                    f"CSeq: 2\r\n"
                    f"User-Agent: SafeVision/4.0\r\n"
                    f"Accept: application/sdp\r\n"
                    f"Authorization: {auth_header}\r\n\r\n"
                )
                sock.sendall(req2.encode("utf-8"))
                resp2 = sock.recv(4096).decode("utf-8", errors="ignore")
                line2 = resp2.splitlines()[0] if resp2.splitlines() else ""
                if " 200 " in line2:
                    return 200, "OK"
                if " 404 " in line2:
                    return 404, "Ruta no encontrada"
                if " 401 " in line2:
                    return 401, "No autorizado"
                return 200, line2
        except Exception as e:
            return 0, str(e)

    def _leer_hikvision_isapi_fallback(self, host, user, pwd):
        """
        Fallback por HTTP ISAPI Digest Auth (/ISAPI/Streaming/channels/102/httpPreview o picture)
        en caso de que OpenCV/FFmpeg falle al decodificar el codec RTSP de la cámara.
        """
        if not user:
            return False
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, f"http://{host}/", user, pwd)
        opener = urllib.request.build_opener(
            urllib.request.HTTPDigestAuthHandler(mgr),
            urllib.request.HTTPBasicAuthHandler(mgr)
        )

        # 1) Intentar stream MJPEG continuo ISAPI si el canal lo soporta
        for ch in ("102", "101"):
            url_mjpeg = f"http://{host}/ISAPI/Streaming/channels/{ch}/httpPreview"
            try:
                resp = opener.open(url_mjpeg, timeout=2.5)
                bytes_buf = bytearray()
                self.connected = True
                self.status_msg = f"ISAPI Activo: http://{host} (Ch {ch})"
                t_prev = time.time()
                cnt = 0
                while not self.stopped:
                    chunk = resp.read(16384)
                    if not chunk:
                        break
                    bytes_buf.extend(chunk)
                    b_end = bytes_buf.rfind(b"\xff\xd9")
                    if b_end != -1:
                        a_start = bytes_buf.rfind(b"\xff\xd8", 0, b_end)
                        if a_start != -1:
                            jpg = bytes(bytes_buf[a_start:b_end + 2])
                            del bytes_buf[:b_end + 2]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                img = self._redimensionar_si_necesario(img)
                                with self.lock:
                                    self.frame = img
                                    self._frame_id += 1
                                cnt += 1
                                now = time.time()
                                if now - t_prev >= 1.0:
                                    self.fps_cam = cnt / (now - t_prev)
                                    cnt = 0
                                    t_prev = now
                        else:
                            del bytes_buf[:b_end + 2]
                if cnt > 0:
                    return True
            except Exception:
                pass

        # 2) Intentar captura de snapshots ISAPI rápidos
        for ch in ("102", "101"):
            url_pic = f"http://{host}/ISAPI/Streaming/channels/{ch}/picture"
            try:
                resp = opener.open(url_pic, timeout=2.5)
                jpg = resp.read()
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.connected = True
                    self.status_msg = f"Snapshot ISAPI Activo: {host}"
                    t_prev = time.time()
                    cnt = 0
                    while not self.stopped:
                        img_r = self._redimensionar_si_necesario(img)
                        with self.lock:
                            self.frame = img_r
                            self._frame_id += 1
                        cnt += 1
                        now = time.time()
                        if now - t_prev >= 1.0:
                            self.fps_cam = cnt / (now - t_prev)
                            cnt = 0
                            t_prev = now
                        resp = opener.open(url_pic, timeout=2.0)
                        jpg = resp.read()
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            break
                    return True
            except Exception:
                pass
        return False

    def _esperar_desbloqueo_o_reinicio_camara(self, host, port):
        """
        Espera sin enviar intentos de login (para no extender el bloqueo de la cámara).
        Monitorea si el usuario apaga/enciende la cámara (caída y regreso del puerto TCP)
        o si termina la cuenta regresiva para reconectar automáticamente al instante.
        """
        vio_desconexion = False
        t_last = time.time()
        while not self.stopped and self.unlock_seconds > 0:
            now = time.time()
            dt = now - t_last
            if dt >= 1.0:
                self.unlock_seconds = max(0, self.unlock_seconds - int(dt))
                t_last = now

            mins, secs = divmod(self.unlock_seconds, 60)
            self.status_msg = (
                f"🔒 Cámara {host} en bloqueo temporal ({mins}m {secs:02d}s). "
                f"Desconecta y conecta su corriente para desbloquear ya."
            )

            # Verificar pulso TCP cada 1.5s sin enviar credenciales
            puerto_abierto = False
            try:
                with socket.create_connection((host, port), timeout=0.8):
                    puerto_abierto = True
            except Exception:
                puerto_abierto = False

            if not puerto_abierto:
                vio_desconexion = True
                self.status_msg = f"🔄 Reinicio detectado en cámara {host}... Esperando arranque..."
            elif vio_desconexion and puerto_abierto:
                # La cámara acaba de reiniciar tras desconectar y conectar la corriente
                self.status_msg = f"✅ Cámara {host} reiniciada. Conectando video..."
                time.sleep(1.5)
                self.is_locked_error = False
                self.unlock_seconds = 0
                return

            time.sleep(1.0)

        self.is_locked_error = False
        self.unlock_seconds = 0

    def _worker(self):
        target = self.normalizar_url(self.raw_input)

        # ==================== CASO 1: WEBCAM USB LOCAL ====================
        if isinstance(target, int):
            while not self.stopped:
                self.status_msg = f"Abriendo Webcam USB #{target}..."
                cap, first_frm = self._abrir_webcam_usb(target)
                if cap is None:
                    self.connected = False
                    self.status_msg = f"Webcam USB #{target} no disponible"
                    time.sleep(1.5)
                    continue

                self.connected = True
                self.status_msg = f"Webcam USB #{target} activa"
                with self.lock:
                    self.frame = self._redimensionar_si_necesario(first_frm)
                    self._frame_id += 1

                t_prev = time.time()
                cnt = 0
                fallos = 0

                while not self.stopped:
                    ret, frm = cap.read()
                    if ret and frm is not None:
                        fallos = 0
                        frm = self._redimensionar_si_necesario(frm)
                        with self.lock:
                            self.frame = frm
                            self._frame_id += 1
                        cnt += 1
                        now = time.time()
                        if now - t_prev >= 1.0:
                            self.fps_cam = cnt / (now - t_prev)
                            cnt = 0
                            t_prev = now
                    else:
                        fallos += 1
                        if fallos > 25:
                            self.connected = False
                            break
                        time.sleep(0.01)
                cap.release()
            return

        # Si el usuario ingresó solo la IP (ej. "192.168.1.64" -> "http://192.168.1.64/video"),
        # comprobar si es una cámara Ethernet RTSP (puerto 554 u 8000 abierto)
        clean_target, user, pwd = self._extraer_credenciales_de_url(str(target))
        if str(target).startswith("http://"):
            host_chk, port_chk, path_chk = self._parsear_host_puerto_ruta(clean_target, 80, "/video")
            if port_chk in (80, 8000) and path_chk in ("/video", "/"):
                try:
                    with socket.create_connection((host_chk, 554), timeout=0.6):
                        target = self._construir_rtsp_con_auth(
                            host_chk, 554, "/Streaming/Channels/102", user, pwd
                        )
                        clean_target, user, pwd = self._extraer_credenciales_de_url(target)
                except Exception:
                    pass

        # ==================== CASO 2: CÁMARA RTSP / ETHERNET ====================
        if str(target).startswith("rtsp://"):
            host, port, req_path = self._parsear_host_puerto_ruta(clean_target, 554, "/Streaming/Channels/102")
            rutas_candidatas = [
                req_path,
                "/Streaming/Channels/102",
                "/Streaming/Channels/101",
                "/h264Preview_01_sub",
                "/cam/realmonitor?channel=1&subtype=1",
                "/live/ch0",
                "/stream2",
                "/stream1",
            ]
            vistos_r = set()
            rutas_rtsp = [r for r in rutas_candidatas if not (r in vistos_r or vistos_r.add(r))]

            credenciales_rechazadas = None

            while not self.stopped:
                # Si las credenciales actuales ya fueron rechazadas (y no han cambiado), no golpear la cámara
                if self.auth_rejected and credenciales_rechazadas == (user, pwd):
                    time.sleep(1.0)
                    continue

                self.status_msg = f"Verificando conexión Ethernet {host}:{port}..."
                ruta_valida = None
                codigo_final = 0

                for r_path in rutas_rtsp:
                    if self.stopped:
                        break
                    code, _ = self._probar_rtsp_describe(host, port, r_path, user, pwd, timeout=2.0)
                    codigo_final = code
                    if code == 200:
                        ruta_valida = r_path
                        break
                    elif code in (0, 401):
                        # Si no hay red (0) o falló auth (401), no probar más rutas para no sumar intentos fallidos
                        break

                if self.stopped:
                    break

                if codigo_final == 0:
                    self.connected = False
                    self.status_msg = f"Sin respuesta en {host}:{port} (Verifica cable Ethernet e IP)"
                    time.sleep(2.0)
                    continue

                if codigo_final == 401:
                    self.connected = False
                    self.requires_auth_error = True
                    self.auth_rejected = True
                    credenciales_rechazadas = (user, pwd)
                    if not user or not pwd:
                        self.status_msg = f"🔒 Cámara {host} requiere Usuario y Contraseña (Ve a Gestor de Cámaras)"
                    else:
                        self.status_msg = (
                            f"🔒 Contraseña incorrecta para '{user}' en {host} (Error 401). "
                            f"Reintentos pausados para no bloquear la cámara."
                        )
                    continue

                # Si llegamos aquí, la autenticación RTSP fue exitosa (200 OK o ruta alternativa)
                self.requires_auth_error = False
                self.auth_rejected = False
                self.is_locked_error = False
                if not ruta_valida:
                    ruta_valida = req_path

                # Probar abrir stream RTSP con OpenCV FFmpeg (intentando ruta validada y luego alternativa 101/102)
                orden_abrir = [ruta_valida]
                for alt_p in ("/Streaming/Channels/102", "/Streaming/Channels/101"):
                    if alt_p not in orden_abrir:
                        orden_abrir.append(alt_p)

                abrio_alguno = False
                for p_try in orden_abrir:
                    if self.stopped:
                        break
                    url_full = self._construir_rtsp_con_auth(host, port, p_try, user, pwd)
                    self.status_msg = f"Abriendo video RTSP {host}:{port}{p_try}..."
                    cap = cv2.VideoCapture(url_full, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        ret, first_frm = cap.read()
                        if ret and first_frm is not None and first_frm.size > 0:
                            abrio_alguno = True
                            self.connected = True
                            self.status_msg = f"RTSP Activo: rtsp://{host}:{port}{p_try}"
                            with self.lock:
                                self.frame = self._redimensionar_si_necesario(first_frm)
                                self._frame_id += 1
                            t_prev = time.time()
                            cnt = 1
                            fallos = 0
                            while not self.stopped:
                                ret, frm = cap.read()
                                if not ret or frm is None:
                                    fallos += 1
                                    if fallos > 15:
                                        self.connected = False
                                        break
                                    time.sleep(0.01)
                                    continue
                                fallos = 0
                                frm = self._redimensionar_si_necesario(frm)
                                with self.lock:
                                    self.frame = frm
                                    self._frame_id += 1
                                cnt += 1
                                now = time.time()
                                if now - t_prev >= 1.0:
                                    self.fps_cam = cnt / (now - t_prev)
                                    cnt = 0
                                    t_prev = now
                        cap.release()
                        if abrio_alguno:
                            break

                # Si RTSP autenticó pero FFmpeg no pudo decodificar, usar fallback HTTP ISAPI
                if not abrio_alguno and not self.stopped:
                    self._leer_hikvision_isapi_fallback(host, user, pwd)

                if not self.stopped:
                    self.connected = False
                    self.status_msg = f"Reconectando cámara Ethernet {host}..."
                    time.sleep(1.5)
            return

        # ==================== CASO 3: CÁMARA HTTP MJPEG / WI-FI ====================
        base_url = clean_target
        for suf in ("/video", "/mjpegfeed", "/shot.jpg", "/videofeed", "/axis-cgi/mjpg/video.cgi"):
            if base_url.endswith(suf):
                base_url = base_url[:-len(suf)]
                break

        candidatos = [
            clean_target,
            base_url + "/video",
            base_url + "/videofeed",
            base_url + "/mjpegfeed",
        ]
        vistos = set()
        urls_probar = [u for u in candidatos if not (u in vistos or vistos.add(u))]

        while not self.stopped:
            conectado_alguna = False
            for url in urls_probar:
                if self.stopped:
                    break
                self.status_msg = f"Conectando a {url}..."
                exito = self._leer_stream_http_mjpeg(url, user, pwd)
                if exito or self.requires_auth_error:
                    conectado_alguna = True
                    break

            if not conectado_alguna and not self.stopped and not self.requires_auth_error:
                cap = cv2.VideoCapture(target)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened():
                    self.connected = True
                    self.status_msg = f"Conectado: {clean_target}"
                    while not self.stopped:
                        ret, frm = cap.read()
                        if not ret or frm is None:
                            self.connected = False
                            break
                        frm = self._redimensionar_si_necesario(frm)
                        with self.lock:
                            self.frame = frm
                            self._frame_id += 1
                    cap.release()

            if not self.stopped:
                self.connected = False
                if self.requires_auth_error:
                    self.status_msg = "🔒 Cámara requiere Usuario y Contraseña (401)"
                    time.sleep(2.5)
                else:
                    self.status_msg = f"Sin respuesta de {base_url}. Reintentando..."
                    time.sleep(1.2)

    def _leer_stream_http_mjpeg(self, url, user="", pwd=""):
        try:
            headers = {'User-Agent': 'SafeVision/4.0'}
            if user:
                token = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
                headers['Authorization'] = f"Basic {token}"

            req = urllib.request.Request(url, headers=headers)
            stream = urllib.request.urlopen(req, timeout=3.5)
            self.requires_auth_error = False

            raw_sock = None
            try:
                raw_sock = stream.fp.raw._sock
                raw_sock.setblocking(False)
            except Exception:
                raw_sock = None

            bytes_buf = bytearray()
            self.connected = True
            self.status_msg = f"Conectado: {url}"
            t_prev = time.time()
            cnt = 0

            while not self.stopped:
                if raw_sock is not None:
                    leido_algo = False
                    while True:
                        try:
                            chunk = raw_sock.recv(65536)
                            if not chunk:
                                return False
                            bytes_buf.extend(chunk)
                            leido_algo = True
                            if len(bytes_buf) > 600000:
                                break
                        except BlockingIOError:
                            break
                        except socket.timeout:
                            break
                        except Exception:
                            return False
                    if not leido_algo:
                        time.sleep(0.002)
                        continue
                else:
                    chunk = stream.read(32768)
                    if not chunk:
                        break
                    bytes_buf.extend(chunk)

                b_end = bytes_buf.rfind(b'\xff\xd9')
                if b_end != -1:
                    a_start = bytes_buf.rfind(b'\xff\xd8', 0, b_end)
                    if a_start != -1:
                        jpg = bytes(bytes_buf[a_start:b_end + 2])
                        del bytes_buf[:b_end + 2]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            img = self._redimensionar_si_necesario(img)
                            with self.lock:
                                self.frame = img
                                self._frame_id += 1
                            cnt += 1
                            now = time.time()
                            if now - t_prev >= 1.0:
                                self.fps_cam = cnt / (now - t_prev)
                                cnt = 0
                                t_prev = now
                    else:
                        del bytes_buf[:b_end + 2]

                if len(bytes_buf) > 400000:
                    ultimo_inicio = bytes_buf.rfind(b'\xff\xd8')
                    if ultimo_inicio != -1:
                        del bytes_buf[:ultimo_inicio]
                    else:
                        bytes_buf.clear()
            return True
        except urllib.error.HTTPError as he:
            if he.code == 401:
                self.requires_auth_error = True
            return False
        except Exception:
            return False

    def read(self):
        with self.lock:
            return self.frame, self._frame_id

    def stop(self):
        self.stopped = True

    # Métodos de compatibilidad para scripts secundarios (ej. line_crossing.py)
    def leer(self):
        with self.lock:
            frm = self.frame.copy() if self.frame is not None else None
        return frm, self.connected, self.status_msg, str(self.normalizar_url(self.raw_input))

    def cambiar_fuente(self, nueva_url):
        self.stop()
        self.__init__(nueva_url, max_width=self.max_width, username=self.username, password=self.password)

    def detener(self):
        self.stop()


class CameraPool:
    """
    Administra instancias compartidas de SmartCamaraIP para que si la Cámara Principal
    y el Monitor de 4 Cámaras usan la misma cámara (USB o IP), compartan el mismo
    hilo de lectura sin duplicar consumo de red/CPU ni bloquear puertos USB.
    """
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()

    @staticmethod
    def _clave(url_or_index):
        return str(SmartCamaraIP.normalizar_url(url_or_index)).strip()

    def obtener_o_crear(self, url_or_index, max_width=640, username="", password=""):
        if url_or_index is None or str(url_or_index).strip() in ("", "none", "off", "-- Desactivada --"):
            return None
        k = self._clave(url_or_index)
        with self.lock:
            cam = self.streams.get(k)
            if cam is not None and not cam.stopped:
                # Si cambiaron credenciales o estaba rechazada con credenciales distintas, recrear
                if (username and username != cam.username) or (password and password != cam.password):
                    cam.stop()
                else:
                    if max_width > cam.max_width:
                        cam.max_width = max_width
                    return cam
            nueva = SmartCamaraIP(url_or_index, max_width=max_width, username=username, password=password)
            self.streams[k] = nueva
            return nueva

    def reconectar(self, url_or_index, max_width=640, username="", password=""):
        k = self._clave(url_or_index)
        with self.lock:
            vieja = self.streams.pop(k, None)
            if vieja is not None:
                vieja.stop()
            nueva = SmartCamaraIP(url_or_index, max_width=max_width, username=username, password=password)
            self.streams[k] = nueva
            return nueva

    def limpiar_no_usadas(self, urls_activas):
        claves_activas = {self._clave(u) for u in urls_activas if u and str(u).strip() not in ("", "none", "off", "-- Desactivada --")}
        with self.lock:
            para_borrar = [k for k in self.streams if k not in claves_activas]
            for k in para_borrar:
                cam = self.streams.pop(k, None)
                if cam is not None:
                    cam.stop()

    def detener_todas(self):
        with self.lock:
            for cam in self.streams.values():
                cam.stop()
            self.streams.clear()


class EscanerRedYWebcams:
    """
    Escáner inteligente de:
    1. Webcams USB e integradas (índices 1..4 con DSHOW + MSMF y nombres WMI reales).
    2. Cámaras en Red Local (Ethernet por cable, Wi-Fi, Hotspot):
       - Descubrimiento SADP Multicast en todas las tarjetas (Hikvision / HiLook / Ezviz)
       - Descubrimiento ONVIF WS-Discovery Multicast en todas las tarjetas de red
       - Tabla ARP + Escaneo completo de subredes Ethernet y Wi-Fi (/24)
       - Puertos: 8080 (IP Webcam), 4747 (DroidCam), 8081 (MJPEG), 554/8554 (RTSP), 8000/8899/37777 (ONVIF/Hikvision/Dahua), 80 (HTTP)
    """
    @staticmethod
    def obtener_nombres_webcams_windows():
        nombres = []
        if os.name != "nt":
            return nombres
        try:
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_PnPEntity -Filter \"PNPClass='Camera' OR PNPClass='Image'\" | Select-Object -ExpandProperty Name"
            ]
            out = subprocess.check_output(
                cmd, timeout=2.5,
                creationflags=subprocess.CREATE_NO_WINDOW
            ).decode("utf-8", errors="ignore")
            for line in out.splitlines():
                line = line.strip()
                if line:
                    nombres.append(line)
        except Exception:
            pass
        return nombres

    @staticmethod
    def obtener_subredes_y_arp(subred_extra=""):
        """
        Devuelve:
        - arp_ips: conjunto de IPs activas en ARP (respuesta inmediata)
        - subredes: lista priorizada de prefijos 'A.B.C' (primero Ethernet físico y Wi-Fi, dejando VirtualBox al final)
        - ips_propias: conjunto de IPs de esta PC
        """
        arp_ips = set()
        subredes_ethernet = []
        subredes_wifi = []
        subredes_virtual = []
        ips_propias = set()

        # 1. Leer todas las interfaces IPv4 de ipconfig clasificando adaptadores físicos vs virtuales
        try:
            out_ip = subprocess.check_output(
                ["ipconfig"], timeout=2.5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ).decode("utf-8", errors="ignore")

            bloques = re.split(r"\r?\n(?=\S.*:)", out_ip)
            for b in bloques:
                b_low = b.lower()
                es_virtual = any(v in b_low for v in ("virtualbox", "vmware", "vethernet", "wsl", "loopback", "bluetooth", "56.1"))
                es_ethernet = ("ethernet" in b_low) and not es_virtual
                for line in b.splitlines():
                    l_low = line.lower()
                    if "ipv4" in l_low or "ip address" in l_low or "dirección ip" in l_low or "direccion ip" in l_low:
                        for match in re.findall(r"(\d+\.\d+\.\d+\.\d+)", line):
                            if match.startswith(("192.168.", "10.", "172.")) and not match.endswith((".255", ".0")):
                                ips_propias.add(match)
                                pref = ".".join(match.split(".")[:3])
                                if pref == "192.168.56" or es_virtual:
                                    if pref not in subredes_virtual:
                                        subredes_virtual.append(pref)
                                elif es_ethernet:
                                    if pref not in subredes_ethernet:
                                        subredes_ethernet.append(pref)
                                else:
                                    if pref not in subredes_wifi:
                                        subredes_wifi.append(pref)
        except Exception:
            pass

        # 2. Detectar IP principal de salida mediante socket UDP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_pri = s.getsockname()[0]
            s.close()
            if ip_pri and not ip_pri.startswith("127."):
                ips_propias.add(ip_pri)
                pref = ".".join(ip_pri.split(".")[:3])
                if pref not in subredes_ethernet and pref not in subredes_wifi:
                    subredes_wifi.append(pref)
        except Exception:
            pass

        # 3. Leer tabla ARP completa (todas las interfaces Ethernet y Wi-Fi)
        try:
            out_arp = subprocess.check_output(
                ["arp", "-a"], timeout=2.0,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ).decode("utf-8", errors="ignore")
            for line in out_arp.splitlines():
                partes = line.split()
                if len(partes) >= 2 and partes[0].count(".") == 3:
                    ip = partes[0]
                    if (
                        not ip.endswith(".255")
                        and not ip.startswith(("224.", "239.", "127.", "255."))
                        and ip not in ips_propias
                    ):
                        arp_ips.add(ip)
                        pref = ".".join(ip.split(".")[:3])
                        if pref != "192.168.56" and pref not in subredes_ethernet and pref not in subredes_wifi:
                            subredes_ethernet.append(pref)
        except Exception:
            pass

        # Orden de prioridad: primero Ethernet por cable, luego Wi-Fi/Hotspot, al final Virtuales
        subredes = []
        for lst in (subredes_ethernet, subredes_wifi, subredes_virtual):
            for p in lst:
                if p not in subredes:
                    subredes.append(p)

        if subred_extra:
            s_clean = subred_extra.strip()
            if s_clean.count(".") == 3:
                arp_ips.add(s_clean)
                pref = ".".join(s_clean.split(".")[:3])
                if pref not in subredes:
                    subredes.insert(0, pref)
            elif s_clean.count(".") == 2:
                if s_clean not in subredes:
                    subredes.insert(0, s_clean)

        return arp_ips, subredes, ips_propias

    @staticmethod
    def descubrir_sadp_hikvision(ips_locales=None, timeout=0.65):
        """
        Envía sonda UDP Multicast SADP (239.255.255.250:37020) en TODAS las tarjetas de red
        (incluyendo Ethernet por cable) para descubrir cámaras Hikvision / HiLook / Ezviz aunque ONVIF esté apagado.
        """
        encontradas_sadp = {}
        probe_xml = (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid><Types>inquiry</Types></Probe>'
        ).encode("utf-8")

        bind_ips = [""]
        if ips_locales:
            for ip_l in ips_locales:
                if ip_l and not ip_l.endswith((".1", ".255", ".0")) or ip_l.startswith("192.168."):
                    if ip_l not in bind_ips:
                        bind_ips.append(ip_l)

        socks = []
        for b_ip in bind_ips:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                if b_ip:
                    s.bind((b_ip, 0))
                s.sendto(probe_xml, ("239.255.255.250", 37020))
                s.setblocking(False)
                socks.append(s)
            except Exception:
                pass

        t0 = time.time()
        while time.time() - t0 < timeout:
            recibio_algo = False
            for s in socks:
                try:
                    data, addr = s.recvfrom(8192)
                    recibio_algo = True
                    xml_str = data.decode("utf-8", errors="ignore")
                    m_ip = re.search(r"<IPv4Address>([^<]+)</IPv4Address>", xml_str)
                    ip_cam = m_ip.group(1).strip() if m_ip else (addr[0] if addr else "")
                    if not ip_cam:
                        continue
                    m_desc = re.search(r"<DeviceDescription>([^<]+)</DeviceDescription>", xml_str)
                    modelo = m_desc.group(1).strip() if m_desc else "Hikvision/HiLook"
                    m_mac = re.search(r"<MAC>([^<]+)</MAC>", xml_str)
                    mac = m_mac.group(1).strip() if m_mac else ""
                    encontradas_sadp[ip_cam] = {
                        "type": "rtsp",
                        "icono": "📹",
                        "name": f"Cámara Ethernet {modelo} ({ip_cam})",
                        "host": ip_cam,
                        "port": "554",
                        "stream_path": "/Streaming/Channels/102",
                        "url": f"rtsp://{ip_cam}:554/Streaming/Channels/102",
                        "requires_auth": True,
                        "detalle": f"🟢 Detectada por cable Ethernet ({modelo} · MAC {mac}) — RTSP Puerto 554"
                    }
                except BlockingIOError:
                    continue
                except Exception:
                    continue
            if not recibio_algo:
                time.sleep(0.02)

        for s in socks:
            try:
                s.close()
            except Exception:
                pass

        return encontradas_sadp

    @staticmethod
    def descubrir_onvif_multicast(ips_locales=None, timeout=0.65):
        """Envía sonda UDP WS-Discovery (239.255.255.250:3702) en todas las tarjetas Ethernet/Wi-Fi."""
        ips_onvif = set()
        probe_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
            'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
            'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
            '<e:Header><w:MessageID>uuid:safevision-probe-01</w:MessageID>'
            '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
            '<w:Action a:mustUnderstand="true" xmlns:a="http://www.w3.org/2005/04/discovery/Probe</w:Action>'
            '</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>'
        ).encode("utf-8")

        bind_ips = [""]
        if ips_locales:
            for ip_l in ips_locales:
                if ip_l not in bind_ips:
                    bind_ips.append(ip_l)

        socks = []
        for b_ip in bind_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(0.15)
                if b_ip:
                    sock.bind((b_ip, 0))
                sock.sendto(probe_xml, ("239.255.255.250", 3702))
                socks.append(sock)
            except Exception:
                pass

        t0 = time.time()
        while time.time() - t0 < timeout:
            for sock in socks:
                try:
                    _, addr = sock.recvfrom(4096)
                    if addr and addr[0]:
                        ips_onvif.add(addr[0])
                except socket.timeout:
                    continue
                except Exception:
                    continue

        for sock in socks:
            try:
                sock.close()
            except Exception:
                pass
        return ips_onvif

    @staticmethod
    def verificar_host_puerto(ip, port, timeout=0.35):
        """
        Comprueba si ip:port está abierto y verifica si es una cámara HTTP/RTSP/ONVIF,
        detectando automáticamente si pide Usuario y Contraseña o si está libre.
        """
        try:
            with socket.create_connection((ip, port), timeout=timeout) as sock:
                sock.settimeout(0.6)
                # 1. Puerto RTSP (554 / 8554)
                if port in (554, 8554):
                    req = f"DESCRIBE rtsp://{ip}:{port}/ RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: SafeVision\r\n\r\n".encode("ascii")
                    try:
                        sock.sendall(req)
                        resp = sock.recv(512).decode("utf-8", errors="ignore")
                    except Exception:
                        resp = ""
                    pide_auth = ("401 Unauthorized" in resp) or ("WWW-Authenticate" in resp) or (resp == "")
                    return {
                        "type": "rtsp",
                        "icono": "📹",
                        "name": f"Cámara Ethernet RTSP ({ip}:{port})",
                        "host": ip,
                        "port": str(port),
                        "stream_path": "/Streaming/Channels/102",
                        "url": f"rtsp://{ip}:{port}/Streaming/Channels/102",
                        "requires_auth": pide_auth,
                        "detalle": "🔒 Requiere Usuario/Contraseña (RTSP)" if pide_auth else "🟢 Stream RTSP Libre (Sin contraseña)"
                    }

                # 2. Puertos ONVIF / Cámaras IP propietarias (8000 Hikvision, 8899 ONVIF, 37777 Dahua)
                if port in (8000, 8899, 37777):
                    marca = "Hikvision" if port == 8000 else ("Dahua" if port == 37777 else "ONVIF")
                    ruta_sug = "/Streaming/Channels/102" if port == 8000 else (
                        "/cam/realmonitor?channel=1&subtype=1" if port == 37777 else "/live/ch0"
                    )
                    return {
                        "type": "rtsp",
                        "icono": "🛡️",
                        "name": f"Cámara {marca} Ethernet/Wi-Fi ({ip})",
                        "host": ip,
                        "port": "554",
                        "stream_path": ruta_sug,
                        "url": f"rtsp://{ip}:554{ruta_sug}",
                        "requires_auth": True,
                        "detalle": f"🔒 Detectada en puerto {port} ({marca}) — Usa Usuario/Contraseña RTSP"
                    }

            # 3. Puertos HTTP (8080 IP Webcam, 4747 DroidCam, 8081 Motion, 80 Web)
            rutas_http = ["/video", "/mjpegfeed", "/videofeed", "/"] if port != 80 else ["/video", "/mjpg/video.mjpg", "/doc/page/login.asp", "/"]
            for ruta in rutas_http:
                url_test = f"http://{ip}:{port}{ruta}"
                try:
                    req = urllib.request.Request(url_test, headers={"User-Agent": "SafeVision/4.0"})
                    with urllib.request.urlopen(req, timeout=0.65) as resp:
                        ctype = str(resp.headers.get("Content-Type", "")).lower()
                        server_hdr = str(resp.headers.get("Server", "")).lower()
                        if (
                            "multipart" in ctype
                            or "image/jpeg" in ctype
                            or "mjpeg" in ctype
                            or port in (8080, 4747, 8081)
                        ):
                            etiqueta = "DroidCam Celular" if port == 4747 else (
                                "IP Webcam Celular" if port == 8080 else f"Cámara HTTP ({ip}:{port})"
                            )
                            url_final = f"http://{ip}:{port}/video" if ruta == "/" and port in (8080, 4747) else url_test
                            return {
                                "type": "http",
                                "icono": "📱" if port in (8080, 4747) else "🌐",
                                "name": f"{etiqueta} ({ip}:{port})",
                                "host": ip,
                                "port": str(port),
                                "stream_path": "/video" if port in (8080, 4747) else ruta,
                                "url": url_final,
                                "requires_auth": False,
                                "detalle": f"🟢 Libre / Sin contraseña — Lista para conectar ({url_final})"
                            }
                        if port == 80 and (
                            ruta == "/doc/page/login.asp"
                            or any(k in server_hdr for k in ("hikvision", "dahua", "ipcam", "camera", "boa", "goahead", "lighttpd", "webserver"))
                        ):
                            return {
                                "type": "rtsp",
                                "icono": "📹",
                                "name": f"Cámara IP Ethernet ({ip})",
                                "host": ip,
                                "port": "554",
                                "stream_path": "/Streaming/Channels/102",
                                "url": f"rtsp://{ip}:554/Streaming/Channels/102",
                                "requires_auth": True,
                                "detalle": f"🔒 Interfaz de cámara detectada en {ip}:80 (RTSP :554)"
                            }
                except urllib.error.HTTPError as he:
                    if he.code == 401:
                        return {
                            "type": "http",
                            "icono": "🔒",
                            "name": f"Cámara IP Protegida ({ip}:{port})",
                            "host": ip,
                            "port": str(port),
                            "stream_path": "/video",
                            "url": f"http://{ip}:{port}/video",
                            "requires_auth": True,
                            "detalle": f"🔒 Requiere Usuario y Contraseña (HTTP 401 en {ip}:{port})"
                        }
                except Exception:
                    continue
        except Exception:
            return None
        return None

    @classmethod
    def escanear_todo(cls, camera_pool=None, subred_extra="", callback_progreso=None):
        """
        Escanea Cámaras en Red (Ethernet, Wi-Fi, Hotspot, SADP, ONVIF, RTSP, HTTP) y
        Webcams USB externas (índices 1..4), SIN encender nunca la cámara integrada de la laptop (índice 0).
        """
        encontradas = []

        # --- PASO 1: Escanear SOLO Webcams USB Externas (índices 1 a 4, NUNCA el 0 de la laptop) ---
        if callback_progreso:
            callback_progreso("Buscando Webcams USB externas (puertos 1 a 4)...")

        nombres_win = cls.obtener_nombres_webcams_windows()
        indices_activos = set()
        if camera_pool is not None:
            with camera_pool.lock:
                for k, cam_obj in camera_pool.streams.items():
                    if k.isdigit() and cam_obj.connected:
                        indices_activos.add(int(k))

        for idx in range(1, 5):
            nombre_hw = nombres_win[idx] if idx < len(nombres_win) else f"Webcam USB Externa #{idx}"
            if idx in indices_activos:
                encontradas.append({
                    "type": "usb",
                    "icono": "🎥",
                    "name": f"{nombre_hw} (USB #{idx})",
                    "host": str(idx),
                    "port": "",
                    "stream_path": "",
                    "url": str(idx),
                    "requires_auth": False,
                    "detalle": f"🟢 Activa actualmente en SafeVision (Índice USB {idx})"
                })
                continue

            abierta = False
            res_str = ""
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_ANY]
            for b in backends:
                try:
                    cap = cv2.VideoCapture(idx, b)
                    if cap.isOpened():
                        for _ in range(3):
                            ret, frm = cap.read()
                            if ret and frm is not None and frm.size > 0:
                                h, w = frm.shape[:2]
                                res_str = f"{w}x{h}"
                                abierta = True
                                break
                            time.sleep(0.02)
                    cap.release()
                    if abierta:
                        break
                except Exception:
                    pass

            if abierta:
                encontradas.append({
                    "type": "usb",
                    "icono": "🎥",
                    "name": f"{nombre_hw} (USB #{idx})",
                    "host": str(idx),
                    "port": "",
                    "stream_path": "",
                    "url": str(idx),
                    "requires_auth": False,
                    "detalle": f"🟢 Webcam USB externa lista — Resolución: {res_str}"
                })

        # --- PASO 2: Descubrimiento SADP (Hikvision/HiLook) + ONVIF Multicast + Subredes Ethernet / Wi-Fi ---
        arp_ips, subredes, ips_propias = cls.obtener_subredes_y_arp(subred_extra)
        if callback_progreso:
            redes_txt = ", ".join(f"{s}.x" for s in subredes[:4]) if subredes else "Red Local"
            callback_progreso(f"Escaneando Ethernet y Wi-Fi ({redes_txt}) por SADP / ONVIF / RTSP / HTTP...")

        por_ip = {}

        # 2.1 Sonda SADP Hikvision/HiLook en todas las interfaces (encuentra cámaras Ethernet en <100ms)
        sadp_map = cls.descubrir_sadp_hikvision(ips_locales=ips_propias, timeout=0.60)
        for ip_s, info_s in sadp_map.items():
            if ip_s not in ips_propias:
                arp_ips.add(ip_s)
                por_ip[ip_s] = info_s

        # 2.2 Sonda ONVIF Multicast en todas las interfaces
        ips_onvif = cls.descubrir_onvif_multicast(ips_locales=ips_propias, timeout=0.55)
        for ip_o in ips_onvif:
            if ip_o not in ips_propias:
                arp_ips.add(ip_o)

        # Construir lista priorizada de IPs: primero ARP + SADP + ONVIF, luego barrido 1..254 de hasta 4 subredes
        ips_prioritarias = list(arp_ips)
        ips_barrido = []
        for pref in subredes[:4]:
            for host_i in range(1, 255):
                cand = f"{pref}.{host_i}"
                if cand not in arp_ips and cand not in ips_propias:
                    ips_barrido.append(cand)

        puertos_rapidos = (8080, 4747, 554, 8081, 8000, 8899, 80)
        puertos_barrido = (554, 8000, 8080, 4747, 8081)

        tareas = []
        vistas_ip_port = set()

        with ThreadPoolExecutor(max_workers=96) as executor:
            for ip in ips_prioritarias:
                for p in puertos_rapidos:
                    vistas_ip_port.add((ip, p))
                    tareas.append(executor.submit(cls.verificar_host_puerto, ip, p, 0.40))

            for ip in ips_barrido:
                for p in puertos_barrido:
                    if (ip, p) not in vistas_ip_port:
                        tareas.append(executor.submit(cls.verificar_host_puerto, ip, p, 0.22))

            for fut in as_completed(tareas):
                res = fut.result()
                if res is not None:
                    ip_h = res["host"]
                    if ip_h not in por_ip:
                        por_ip[ip_h] = res
                    else:
                        # Preferir nombre más descriptivo (ej. SADP o Hikvision 8000 sobre RTSP genérico)
                        actual = por_ip[ip_h]
                        if "IPC-" in actual["name"] or "Hikvision" in actual["name"]:
                            continue
                        if "Hikvision" in res["name"] or "DroidCam" in res["name"] or "IP Webcam" in res["name"]:
                            por_ip[ip_h] = res

        for cam_net in por_ip.values():
            encontradas.append(cam_net)

        return encontradas
