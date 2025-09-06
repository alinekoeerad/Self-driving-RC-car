#include <WiFi.h>
#include <WiFiUdp.h>

// WiFi credentials
const char* ssid = "LA_D";
const char* password = "1223334444";

// UDP settings
WiFiUDP udp;
unsigned int localUdpPort = 12345;  // Port to listen on
char incomingPacket[255];

// Motor control pins
const int motorLpin1 = 5;
const int motorLpin2 = 18;
const int motorRpin1 = 19;
const int motorRpin2 = 21;

void setup() {
  Serial.begin(115200);

  // Initialize motor control pins
  pinMode(motorLpin1, OUTPUT);
  pinMode(motorLpin2, OUTPUT);
  pinMode(motorRpin1, OUTPUT);
  pinMode(motorRpin2, OUTPUT);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");

  // Start UDP
  udp.begin(localUdpPort);
}

void loop() {
  // Check for incoming UDP packets
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0;  // Null-terminate the string
    }

    // Print the received packet
    Serial.printf("Received packet: %s\n", incomingPacket);

    // Convert the received packet to an integer angle
    int angle = atoi(incomingPacket);

    // Adjust the motors based on the received angle
    adjustMotors(angle);
  }
}

void adjustMotors(int angle) {
  int speedL, speedR;

  if (angle < 90) {
    speedL = map(angle, 0, 90, 0, 255);  
    speedR = 255;  
  } else if (angle > 90) {
    speedL = 255;  
    speedR = map(angle, 90, 180, 255, 0);  
  } else {
    speedL = 255;
    speedR = 255;
  }
 
  analogWrite(motorLpin1, speedL);
  digitalWrite(motorLpin2, LOW);

  analogWrite(motorRpin1, speedR);
  digitalWrite(motorRpin2, LOW);
}
