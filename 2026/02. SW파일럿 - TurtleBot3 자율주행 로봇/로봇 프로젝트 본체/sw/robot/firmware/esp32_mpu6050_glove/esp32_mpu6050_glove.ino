/*
 * ESP32 + MPU6050 Wi-Fi glove transmitter for TurtleBot3.
 *
 * This is an Arduino-compatible port of main/esp32_test.c from
 * esp32_test_code_bundle.pdf. It keeps the original electrical settings,
 * MPU6050 setup, 200-sample gyro calibration, 0.96 complementary filter,
 * CTRL_ROLL axis and eight direction states.
 *
 * Every 50 ms it sends the original message format over UDP 5005:
 *   ROLL:  -1.20 | CTRL_ROLL:   1.20 | PITCH:-18.30 | STATE:FORWARD_RIGHT
 *
 * The ROS receiver announces itself with IMU_DISCOVER_V1 over UDP 5006, so
 * no fixed TurtleBot computer IP is required. The configured IP remains a
 * fallback until the receiver is discovered.
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "wifi_config.h"

constexpr int I2C_SDA_GPIO = 21;
constexpr int I2C_SCL_GPIO = 22;
constexpr uint8_t MPU6050_ADDR = 0x68;
constexpr uint32_t I2C_CLOCK_HZ = 100000;
constexpr uint32_t SERIAL_BAUD = 115200;

constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_WHO_AM_I = 0x75;
constexpr uint8_t MPU6050_WHO_AM_I = 0x68;
constexpr uint8_t MPU6500_WHO_AM_I = 0x70;

constexpr uint16_t UDP_DATA_PORT = 5005;
constexpr uint16_t UDP_DISCOVERY_PORT = 5006;
constexpr char UDP_DISCOVERY_MESSAGE[] = "IMU_DISCOVER_V1";
constexpr uint32_t WIFI_RETRY_INTERVAL_MS = 5000;
constexpr uint32_t OUTPUT_INTERVAL_MS = 50;
constexpr float PITCH_THRESHOLD = 15.0f;
constexpr float ROLL_THRESHOLD = 15.0f;
constexpr float STOP_PITCH_LIMIT = 10.0f;
constexpr float STOP_ROLL_LIMIT = 10.0f;

struct ImuSample {
  float accelX;
  float accelY;
  float accelZ;
  float gyroX;
  float gyroY;
};

WiFiUDP gloveUdp;
IPAddress receiverIp;
bool mpuReady = false;
bool firstSample = true;
float gyroXOffset = 0.0f;
float gyroYOffset = 0.0f;
float rollDegrees = 0.0f;
float pitchDegrees = 0.0f;
uint32_t previousSampleMs = 0;
uint32_t lastOutputMs = 0;
uint32_t lastMpuRetryMs = 0;
uint32_t lastWifiRetryMs = 0;

int16_t combineBytes(uint8_t high, uint8_t low) {
  return static_cast<int16_t>((static_cast<uint16_t>(high) << 8) | low);
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission(true) == 0;
}

bool readRegisters(uint8_t startReg, uint8_t *data, size_t length) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(startReg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(static_cast<int>(MPU6050_ADDR), static_cast<int>(length),
                       true) != length) return false;
  for (size_t index = 0; index < length; ++index) data[index] = Wire.read();
  return true;
}

bool readImu(ImuSample *sample) {
  uint8_t data[14] = {};
  if (!readRegisters(REG_ACCEL_XOUT_H, data, sizeof(data))) return false;
  sample->accelX = static_cast<float>(combineBytes(data[0], data[1])) / 16384.0f;
  sample->accelY = static_cast<float>(combineBytes(data[2], data[3])) / 16384.0f;
  sample->accelZ = static_cast<float>(combineBytes(data[4], data[5])) / 16384.0f;
  sample->gyroX = static_cast<float>(combineBytes(data[8], data[9])) / 131.0f;
  sample->gyroY = static_cast<float>(combineBytes(data[10], data[11])) / 131.0f;
  return true;
}

bool initMpu6050() {
  Wire.begin(I2C_SDA_GPIO, I2C_SCL_GPIO);
  Wire.setClock(I2C_CLOCK_HZ);

  Wire.beginTransmission(MPU6050_ADDR);
  const uint8_t probeResult = Wire.endTransmission(true);
  if (probeResult != 0) {
    Serial.printf("I2C probe 0x68 failed (Wire error %u).\n", probeResult);
    return false;
  }
  uint8_t whoAmI = 0;
  if (!readRegisters(REG_WHO_AM_I, &whoAmI, 1)) {
    Serial.println("MPU6050 WHO_AM_I read failed.");
    return false;
  }
  if (whoAmI != MPU6050_WHO_AM_I && whoAmI != MPU6500_WHO_AM_I) {
    Serial.printf("MPU6050 WHO_AM_I mismatch: 0x%02X.\n", whoAmI);
    return false;
  }
  if (whoAmI == MPU6500_WHO_AM_I) {
    Serial.println("MPU6500-compatible IMU detected; using MPU6050 registers.");
  }
  if (!writeRegister(REG_PWR_MGMT_1, 0x00)) return false;
  delay(100);
  if (!writeRegister(REG_SMPLRT_DIV, 0x07)) return false;
  if (!writeRegister(REG_CONFIG, 0x03)) return false;
  if (!writeRegister(REG_GYRO_CONFIG, 0x00)) return false;
  if (!writeRegister(REG_ACCEL_CONFIG, 0x00)) return false;

  Serial.println("MPU6050 detected at 0x68 (SDA=GPIO21, SCL=GPIO22).");
  Serial.println("Keep glove still for gyro calibration.");
  float gyroXSum = 0.0f;
  float gyroYSum = 0.0f;
  ImuSample sample = {};
  constexpr int calibrationSamples = 200;
  for (int index = 0; index < calibrationSamples; ++index) {
    if (!readImu(&sample)) return false;
    gyroXSum += sample.gyroX;
    gyroYSum += sample.gyroY;
    delay(10);
  }
  gyroXOffset = gyroXSum / calibrationSamples;
  gyroYOffset = gyroYSum / calibrationSamples;
  firstSample = true;
  previousSampleMs = millis();
  Serial.printf("Gyro offset: X=%.3f Y=%.3f\n", gyroXOffset, gyroYOffset);
  Serial.println("Glove IMU streaming ready.");
  return true;
}

const char *directionState(float ctrlRoll, float pitch) {
  if (fabsf(ctrlRoll) < STOP_ROLL_LIMIT && fabsf(pitch) < STOP_PITCH_LIMIT) return "STOP";
  if (pitch <= -PITCH_THRESHOLD && ctrlRoll <= -ROLL_THRESHOLD) return "FORWARD_LEFT";
  if (pitch <= -PITCH_THRESHOLD && ctrlRoll >= ROLL_THRESHOLD) return "FORWARD_RIGHT";
  if (pitch >= PITCH_THRESHOLD && ctrlRoll <= -ROLL_THRESHOLD) return "BACKWARD_LEFT";
  if (pitch >= PITCH_THRESHOLD && ctrlRoll >= ROLL_THRESHOLD) return "BACKWARD_RIGHT";
  if (pitch <= -PITCH_THRESHOLD) return "FORWARD";
  if (pitch >= PITCH_THRESHOLD) return "BACKWARD";
  if (ctrlRoll <= -ROLL_THRESHOLD) return "LEFT";
  if (ctrlRoll >= ROLL_THRESHOLD) return "RIGHT";
  return "STOP";
}

void connectWifi() {
  WiFi.disconnect(false, false);
  WiFi.begin(GLOVE_WIFI_SSID, GLOVE_WIFI_PASSWORD);
  Serial.println("Wi-Fi connecting.");
}

void onWifiEvent(WiFiEvent_t event, WiFiEventInfo_t info) {
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.printf("Wi-Fi disconnected (reason=%d).\n",
                  info.wifi_sta_disconnected.reason);
  }
}

void maintainWifi(uint32_t now) {
  static bool wasConnected = false;
  if (WiFi.status() == WL_CONNECTED) {
    if (!wasConnected) {
      wasConnected = true;
      Serial.print("Wi-Fi connected. ESP32 IP: ");
      Serial.println(WiFi.localIP());
    }
    return;
  }
  wasConnected = false;
  if (now - lastWifiRetryMs >= WIFI_RETRY_INTERVAL_MS) {
    lastWifiRetryMs = now;
    Serial.printf("Wi-Fi not connected (status=%d); retrying.\n", WiFi.status());
    connectWifi();
  }
}

void pollReceiverDiscovery() {
  const int packetSize = gloveUdp.parsePacket();
  if (packetSize <= 0) return;
  char message[64] = {};
  const int count = gloveUdp.read(
      message, min(packetSize, static_cast<int>(sizeof(message) - 1)));
  if (count <= 0) return;
  message[count] = '\0';
  if (strcmp(message, UDP_DISCOVERY_MESSAGE) != 0) return;
  receiverIp = gloveUdp.remoteIP();
  Serial.print("UDP receiver discovered: ");
  Serial.print(receiverIp);
  Serial.print(":");
  Serial.println(UDP_DATA_PORT);
}

void sendOrientation(float roll, float ctrlRoll, float pitch, const char *state) {
  char message[160] = {};
  snprintf(message, sizeof(message),
           "ROLL: %6.2f | CTRL_ROLL: %6.2f | PITCH: %6.2f | STATE:%s",
           roll, ctrlRoll, pitch, state);
  Serial.println(message);
  if (WiFi.status() != WL_CONNECTED) return;
  gloveUdp.beginPacket(receiverIp, UDP_DATA_PORT);
  gloveUdp.print(message);
  gloveUdp.endPacket();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  receiverIp.fromString(GLOVE_UDP_RECEIVER_IP);

  // 장갑은 wifi_config.h에 지정한 한 AP만 사용한다. 이전에 저장된 AP 설정을
  // 지우고 자동 재접속 기능도 끈 뒤, maintainWifi()가 지정 AP로만 재시도한다.
  // 따라서 다른 Wi-Fi가 저장되어 있거나 근처에 있어도 연결 대상으로 쓰지 않는다.
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(false, true);
  WiFi.setAutoReconnect(false);
  WiFi.setSleep(false);
  WiFi.onEvent(onWifiEvent, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  gloveUdp.begin(UDP_DISCOVERY_PORT);
  Serial.println("ESP32 MPU6050 Wi-Fi glove starting.");
  Serial.println("I2C required: SDA=GPIO21, SCL=GPIO22, MPU6050 address=0x68.");
  connectWifi();
}

void loop() {
  const uint32_t now = millis();
  maintainWifi(now);
  if (WiFi.status() == WL_CONNECTED) pollReceiverDiscovery();

  if (!mpuReady) {
    if (now - lastMpuRetryMs >= 1000) {
      lastMpuRetryMs = now;
      mpuReady = initMpu6050();
      if (!mpuReady) {
        Serial.println("MPU6050 not detected: check 3V3, GND, SDA GPIO21, SCL GPIO22.");
      }
    }
    delay(1);
    return;
  }

  ImuSample sample = {};
  if (!readImu(&sample)) {
    Serial.println("MPU6050 read error; wireless stream stopped.");
    mpuReady = false;
    delay(1);
    return;
  }

  const float accelRoll = atan2f(sample.accelY, sample.accelZ) * RAD_TO_DEG;
  const float accelPitch = atan2f(-sample.accelX,
                                  sqrtf(sample.accelY * sample.accelY +
                                        sample.accelZ * sample.accelZ)) * RAD_TO_DEG;
  float dt = static_cast<float>(now - previousSampleMs) / 1000.0f;
  previousSampleMs = now;
  if (dt <= 0.0f || dt > 0.5f) dt = 0.05f;

  if (firstSample) {
    rollDegrees = accelRoll;
    pitchDegrees = accelPitch;
    firstSample = false;
  } else {
    rollDegrees = 0.96f * (rollDegrees + (sample.gyroX - gyroXOffset) * dt) +
                  0.04f * accelRoll;
    pitchDegrees = 0.96f * (pitchDegrees + (sample.gyroY - gyroYOffset) * dt) +
                   0.04f * accelPitch;
  }

  if (now - lastOutputMs >= OUTPUT_INTERVAL_MS) {
    lastOutputMs = now;
    const float ctrlRoll = -rollDegrees;
    sendOrientation(rollDegrees, ctrlRoll, pitchDegrees,
                    directionState(ctrlRoll, pitchDegrees));
  }
  delay(1);
}
