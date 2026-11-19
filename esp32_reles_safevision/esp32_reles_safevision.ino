/*
 * ============================================================================
 * SAFEVISION INDUSTRIAL - FIRMWARE ESP32 (MODO 100% USB SERIAL DIRECTO)
 * ============================================================================
 * Version optimizada exclusivamente para comunicacion por CABLE USB (115200 baud).
 * Sin Wi-Fi para evitar caidas de tension (Brownout) al activar los reles y
 * garantizar respuesta instantanea (< 5 ms).
 *
 * FUNCIONAMIENTO POR ZONAS DE YOLO:
 *   - AREA 1 (ALARMA):           Enciende Rele 1 (Alarma / Sirena)
 *   - AREA 2 (CORTE CORRIENTE):  Enciende Rele 2 (Corta la corriente) + Rele 1 (Alarma) + Rele 4 (Aux)
 *   - ZONAS DESPEJADAS (NORMAL): Apaga Alarma y Corte; Enciende Rele 3 (Luz Verde OK)
 *
 * CONEXION FISICA (ESP32 -> Placa de 4 Reles):
 * ----------------------------------------------------------------------------
 *   ESP32 Pin VIN (5V USB) --->  VCC del modulo de 4 reles
 *   ESP32 Pin GND          --->  GND del modulo de 4 reles
 *   ESP32 GPIO 26          --->  IN1 (Rele 1: Alarma / Sirena)
 *   ESP32 GPIO 27          --->  IN2 (Rele 2: Corte de Corriente)
 *   ESP32 GPIO 14          --->  IN3 (Rele 3: Luz Verde / Sistema OK)
 *   ESP32 GPIO 25          --->  IN4 (Rele 4: Auxiliar / Baliza)
 *
 * COMO PROBAR DESDE EL MONITOR SERIAL DE ARDUINO IDE (115200 baudios):
 *   Escribe cualquiera de estos comandos y presiona Enter:
 *     1  (o STATE:ALARMA)  -> Enciende el Rele 1 (Alarma)
 *     2  (o STATE:CORTE)   -> Enciende el Rele 2 (Corte de Corriente) + Rele 1 (Alarma)
 *     0  (o STATE:NORMAL)  -> Estado Normal (Apaga Alarma y Corte, enciende Rele 3 Verde)
 *     R1:1 / R1:0          -> Enciende / Apaga manualmente el Rele 1
 *     R2:1 / R2:0          -> Enciende / Apaga manualmente el Rele 2
 *     R3:1 / R3:0          -> Enciende / Apaga manualmente el Rele 3
 *     R4:1 / R4:0          -> Enciende / Apaga manualmente el Rele 4
 * ============================================================================
 */

// ================= PINES DE LA PLACA DE 4 RELES ===================
const int PIN_RELE_1_ALARMA   = 26;  // IN1 -> Rele 1: Alarma (Zona 1)
const int PIN_RELE_2_CORTE    = 27;  // IN2 -> Rele 2: Corte de Corriente (Zona 2)
const int PIN_RELE_3_VERDE_OK = 14;  // IN3 -> Rele 3: Luz Verde (Maquina Operando)
const int PIN_RELE_4_AUX      = 25;  // IN4 -> Rele 4: Auxiliar
const int PIN_LED_INTERNO     = 2;   // LED azul integrado en la ESP32

// Los modulos de 4 reles estandar se activan con LOW (0V) y se apagan con HIGH (3.3V).
// Si tu modulo se activa con HIGH, cambia esto a false (o desde la interfaz de Python):
bool RELE_ACTIVO_EN_LOW = true;

// Estado actual de cada rele (true = relé activado/energizado, false = relé en reposo/apagado)
bool estadoRele[4] = {false, false, true, false};
String estadoSistema = "NORMAL";
String bufferSerial = "";

int nivelFisico(bool encender) {
  if (RELE_ACTIVO_EN_LOW) {
    return encender ? LOW : HIGH;
  } else {
    return encender ? HIGH : LOW;
  }
}

void escribirRele(int indice, bool encender) {
  if (indice < 0 || indice > 3) return;
  estadoRele[indice] = encender;

  int pin = PIN_RELE_1_ALARMA;
  if (indice == 0) pin = PIN_RELE_1_ALARMA;
  else if (indice == 1) pin = PIN_RELE_2_CORTE;
  else if (indice == 2) pin = PIN_RELE_3_VERDE_OK;
  else if (indice == 3) pin = PIN_RELE_4_AUX;

  digitalWrite(pin, nivelFisico(encender));
}

void aplicarEstadoSeguridad(String nuevoEstado) {
  nuevoEstado.trim();
  nuevoEstado.toUpperCase();

  if (nuevoEstado == "NORMAL" || nuevoEstado == "SEGURO" || nuevoEstado == "0" || nuevoEstado == "N") {
    estadoSistema = "NORMAL";
    escribirRele(0, false); // Rele 1 (Alarma): APAGADO
    escribirRele(1, false); // Rele 2 (Corte Corriente): EN REPOSO (Corriente pasa normal)
    escribirRele(2, true);  // Rele 3 (Luz Verde OK): ENCENDIDO
    escribirRele(3, false); // Rele 4 (Auxiliar): APAGADO
    digitalWrite(PIN_LED_INTERNO, LOW);
  }
  else if (nuevoEstado == "ALARMA" || nuevoEstado == "ADVERTENCIA" || nuevoEstado == "1" || nuevoEstado == "A") {
    estadoSistema = "ALARMA";
    escribirRele(0, true);  // Rele 1 (Alarma): ENCENDIDO (Persona en Area 1)
    escribirRele(1, false); // Rele 2 (Corte Corriente): EN REPOSO (No corta la corriente)
    escribirRele(2, false); // Rele 3 (Luz Verde OK): APAGADO
    escribirRele(3, false); // Rele 4 (Auxiliar): APAGADO
    digitalWrite(PIN_LED_INTERNO, HIGH);
  }
  else if (nuevoEstado == "CORTE" || nuevoEstado == "PELIGRO" || nuevoEstado == "PARO" || nuevoEstado == "2" || nuevoEstado == "C") {
    estadoSistema = "CORTE";
    escribirRele(0, true);  // Rele 1 (Alarma): ENCENDIDO
    escribirRele(1, true);  // Rele 2 (Corte Corriente): ACTIVADO (Corta la corriente en Area 2)
    escribirRele(2, false); // Rele 3 (Luz Verde OK): APAGADO
    escribirRele(3, true);  // Rele 4 (Auxiliar): ENCENDIDO
    digitalWrite(PIN_LED_INTERNO, HIGH);
  }
  else if (nuevoEstado == "BYPASS" || nuevoEstado == "OFF" || nuevoEstado == "3" || nuevoEstado == "B") {
    estadoSistema = "BYPASS";
    escribirRele(0, false);
    escribirRele(1, false);
    escribirRele(2, false);
    escribirRele(3, false);
    digitalWrite(PIN_LED_INTERNO, LOW);
  }
}

void responderEstadoJSON() {
  Serial.print("{\"ok\":true,\"state\":\"");
  Serial.print(estadoSistema);
  Serial.print("\",\"r1\":");
  Serial.print(estadoRele[0] ? 1 : 0);
  Serial.print(",\"r2\":");
  Serial.print(estadoRele[1] ? 1 : 0);
  Serial.print(",\"r3\":");
  Serial.print(estadoRele[2] ? 1 : 0);
  Serial.print(",\"r4\":");
  Serial.print(estadoRele[3] ? 1 : 0);
  Serial.print(",\"active_low\":");
  Serial.print(RELE_ACTIVO_EN_LOW ? "true" : "false");
  Serial.println("}");
}

void procesarComando(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd == "PING" || cmd == "STATUS" || cmd == "?") {
    responderEstadoJSON();
    return;
  }

  if (cmd.startsWith("STATE:")) {
    aplicarEstadoSeguridad(cmd.substring(6));
    responderEstadoJSON();
    return;
  }

  if (cmd == "NORMAL" || cmd == "ALARMA" || cmd == "CORTE" || cmd == "BYPASS" ||
      cmd == "0" || cmd == "1" || cmd == "2" || cmd == "3" ||
      cmd == "N" || cmd == "A" || cmd == "C" || cmd == "B") {
    aplicarEstadoSeguridad(cmd);
    responderEstadoJSON();
    return;
  }

  if (cmd.startsWith("R") && cmd.length() >= 4 && cmd.charAt(2) == ':') {
    int numRele = cmd.charAt(1) - '1'; // '1'..'4' -> 0..3
    char accion = cmd.charAt(3);
    if (numRele >= 0 && numRele <= 3) {
      bool nuevoVal = (accion == '1') ? true : ((accion == '0') ? false : !estadoRele[numRele]);
      estadoSistema = "MANUAL";
      escribirRele(numRele, nuevoVal);
      digitalWrite(PIN_LED_INTERNO, nuevoVal ? HIGH : LOW);
      responderEstadoJSON();
      return;
    }
  }

  if (cmd == "LOW:1") {
    RELE_ACTIVO_EN_LOW = true;
    aplicarEstadoSeguridad(estadoSistema);
    responderEstadoJSON();
    return;
  }

  if (cmd == "LOW:0") {
    RELE_ACTIVO_EN_LOW = false;
    aplicarEstadoSeguridad(estadoSistema);
    responderEstadoJSON();
    return;
  }

  responderEstadoJSON();
}

void setup() {
  // 1) Pre-cargar niveles de salida ANTES de habilitar OUTPUT para evitar que los reles se disparen al energizar
  digitalWrite(PIN_RELE_1_ALARMA,   nivelFisico(false));
  digitalWrite(PIN_RELE_2_CORTE,    nivelFisico(false));
  digitalWrite(PIN_RELE_3_VERDE_OK, nivelFisico(true));
  digitalWrite(PIN_RELE_4_AUX,      nivelFisico(false));

  pinMode(PIN_RELE_1_ALARMA,   OUTPUT);
  pinMode(PIN_RELE_2_CORTE,    OUTPUT);
  pinMode(PIN_RELE_3_VERDE_OK, OUTPUT);
  pinMode(PIN_RELE_4_AUX,      OUTPUT);
  pinMode(PIN_LED_INTERNO,     OUTPUT);

  aplicarEstadoSeguridad("NORMAL");

  // 2) Iniciar Puerto Serial USB a 115200 baudios
  Serial.begin(115200);
  delay(50);
  responderEstadoJSON();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufferSerial.length() > 0) {
        procesarComando(bufferSerial);
        bufferSerial = "";
      }
    } else {
      bufferSerial += c;
      if (bufferSerial.length() > 80) {
        bufferSerial = "";
      }
    }
  }
}
