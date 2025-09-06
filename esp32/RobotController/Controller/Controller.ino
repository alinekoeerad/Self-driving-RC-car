#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// === 1. WIFI AND SERVER CONFIGURATION ===
const char* ssid = "LA_D";
const char* password = "1223334444";
const char* serverUrl = "http://192.168.137.1:5000/get_command";

// === 2. MOTOR AND LED PINS ===
const int IN1 = 18; // Left Motor
const int IN2 = 19;
const int IN3 = 22; // Right Motor
const int IN4 = 23;
const int LED_PIN = 2;

const long httpTimeout = 100;
unsigned long lastHttpTime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  stopMotors();
  digitalWrite(LED_PIN, LOW); // Start with LED off
  connectToWiFi();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED && (millis() - lastHttpTime > httpTimeout)) {
    getAndProcessCommand();
    lastHttpTime = millis();
  }
}

// --- Motor Control Functions ---
void drive(int leftSpeed, int rightSpeed) {
  leftSpeed = constrain(leftSpeed, 0, 255);
  rightSpeed = constrain(rightSpeed, 0, 255);
  analogWrite(IN1, leftSpeed);
  digitalWrite(IN2, 0);
  analogWrite(IN3, rightSpeed);
  digitalWrite(IN4, 0);
}

void stopMotors() {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 0);
}

void turnLeftOnSpot() {
  digitalWrite(IN1, 0);
  analogWrite(IN2, 200);
  analogWrite(IN3, 200);
  digitalWrite(IN4, 0);
}

void turnRightOnSpot() {
  analogWrite(IN1, 200);
  digitalWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 200);
}


void getAndProcessCommand() {
  HTTPClient http;
  http.begin(serverUrl);
  int httpCode = http.GET();
  if (httpCode > 0) {
    if (httpCode == HTTP_CODE_OK) {
      String payload = http.getString();
      JsonDocument doc;
      deserializeJson(doc, payload);

      // --- NEW LOGIC: Always process the full state ---
      
      // 1. Handle the LED state in every successful request
      if (doc.containsKey("led")) {
        String led_cmd = doc["led"].as<String>();
        if (led_cmd == "LED_ON") {
          digitalWrite(LED_PIN, HIGH);
        } else if (led_cmd == "LED_OFF") {
          digitalWrite(LED_PIN, LOW);
        }
      }

      // 2. Handle the Movement state
      if (doc.containsKey("command")) {
        String command = doc["command"].as<String>();
        if (command == "DRIVE") {
          if (doc.containsKey("payload")) {
            int left = doc["payload"]["left"];
            int right = doc["payload"]["right"];
            drive(left, right);
          }
        } else if (command == "STOP") {
          stopMotors();
        } else if (command == "LEFT") {
          turnLeftOnSpot();
        } else if (command == "RIGHT") {
          turnRightOnSpot();
        }
      }
    }
  } else {
    stopMotors(); // Failsafe if the server can't be reached
  }
  http.end();
}

void connectToWiFi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}