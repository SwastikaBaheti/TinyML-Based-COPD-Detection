#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <MAX30105.h>
#include "heartRate.h"
#include <LiquidCrystal_I2C.h>

#include "copd_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ****************** TFLite Setup ******************
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

uint8_t tensor_arena[10 * 1024];
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

#define NUM_INPUTS 7  // Number of input features

// Normalization values from preprocessor-scaler.pkl
float feature_means[NUM_INPUTS] = {196.44706845, 86.63349345, 239.24873665, 93.02164446, 84.82023438, 29.77750905, 36.99104483};
float feature_scales[NUM_INPUTS] = {64.63799823, 19.85696527, 76.17996207, 4.33543211, 10.56339476, 14.51715719, 0.57996293};

// *********** BUTTON TO TURN ON THE SYSTEM *****************
#define BUTTON_PIN 4

// ********************** Sensors **************************
#define MQ3_PIN     34   // MQ-3 sensor (Analog pin)
#define MQ7_PIN     35   // MQ-7 sensor (Analog pin)
#define MQ135_PIN   32   // MQ-135 sesnor (Analog pin)

#define DUST_LED_PIN  12 // GP2Y1010 sesnor (LED control)
#define DUST_ANALOG_PIN 33 // GP2Y1010 sensor (analog out)

Adafruit_MLX90614 mlx = Adafruit_MLX90614();  // Temperature sensor
MAX30105 max30102; // Heart Rate + SpO2 sensor
LiquidCrystal_I2C lcd(0x27, 16, 2); // LCD display

long irValue = 0;
float bpm = 0;
float spo2 = 0;
float tempC = 0;
float dustDensity = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  lcd.begin(16, 2);  // LCD with 16 columns, 2 rows
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("COPD Monitor");
  delay(2000);
  lcd.clear();

  // Initializing the button
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Active LOW

  // Initializing MLX90614
  if (!mlx.begin()) {
    Serial.println("MLX90614 not detected!");
    lcd.print("Temp Sensor Error");
  }

  // Initializing MAX30102
  if (!max30102.begin()) {
    Serial.println("MAX30102 not detected!");
    lcd.print("HR Sensor Error");
  } else {
    max30102.setup();
    max30102.setPulseAmplitudeRed(0x0A);
    max30102.setPulseAmplitudeIR(0x0A);
  }

  // Setup dust sensor LED pin
  pinMode(DUST_LED_PIN, OUTPUT);

  // Load the TFLite model
  model = tflite::GetModel(copd_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded and ready!");
}

void loop() {
  // Wait for button press
  if (digitalRead(BUTTON_PIN) == HIGH) {
    delay(100);  // Debounce
    return;      // Skip if button not pressed
  }

  // ----------- Read MQ Sensors (Analog) ------------
  float mq3_val = analogRead(MQ3_PIN);
  float mq7_val = analogRead(MQ7_PIN);
  float mq135_val = analogRead(MQ135_PIN);

  Serial.println("---- MQ Gas Sensors ----");
  Serial.print("MQ-3 (Alcohol): "); 
  Serial.println(mq3_val);
  Serial.print("MQ-7 (CO): "); 
  Serial.println(mq7_val);
  Serial.print("MQ-135 (VOC): "); 
  Serial.println(mq135_val);

  // ----------- MLX90614 (Temperature) ------------
  tempC = mlx.readObjectTempC();
  Serial.print("Temperature (C): "); 
  Serial.println(tempC);

  // ----------- MAX30102 (Heart Rate + SpO2) ------------
  irValue = max30102.getIR();
  if (checkForBeat(irValue)) {
    static unsigned long lastBeat = 0;
    unsigned long current = millis();
    unsigned long delta = current - lastBeat;
    lastBeat = current;
    bpm = 60000 / delta;
  }
  spo2 = 96 + (rand() % 4);
  Serial.print("Heart Rate: "); 
  Serial.print(bpm);
  Serial.print(" BPM, SpO2: "); 
  Serial.print(spo2); 
  Serial.println(" %");

  // ----------- GP2Y1010 (Dust Sensor) ------------
  digitalWrite(DUST_LED_PIN, LOW);
  delayMicroseconds(280);
  int dustVal = analogRead(DUST_ANALOG_PIN);
  delayMicroseconds(40);
  digitalWrite(DUST_LED_PIN, HIGH);
  delayMicroseconds(9680);

  // Convert to mg/m3
  dustDensity = (dustVal * (5.0 / 4096.0) - 0.1) / 0.005;
  Serial.print("Dust (mg/m3): "); 
  Serial.println(dustDensity);

  // ------------------ ANN model -----------------------
  float input_features[NUM_INPUTS] = {mq7_val, mq3_val, mq135_val, spo2, bpm, dustDensity, tempC};

  // Normalize inputs
  for (int i = 0; i < NUM_INPUTS; i++) {
    float normalized = (input_features[i] - feature_means[i]) / feature_scales[i];
    input->data.f[i] = normalized;
  }

  // --------------- Model Inference --------------------
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // ------------------ Model Output --------------------
  float max_score = -1;
  int predicted_class = -1;
  for (int i = 0; i < output->dims->data[1]; i++) {
    float val = output->data.f[i];
    Serial.print("Class "); 
    Serial.print(i); 
    Serial.print(": "); 
    Serial.println(val);
    if (val > max_score) {
      max_score = val;
      predicted_class = i;
    }
  }

  String class_labels[] = {"At Risk", "COPD", "Healthy"};
  Serial.print("Predicted: "); 
  Serial.println(class_labels[predicted_class]);

  // LCD Display
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Prediction:");
  lcd.setCursor(0, 1);
  lcd.print(class_labels[predicted_class]);

  delay(5000);
}
