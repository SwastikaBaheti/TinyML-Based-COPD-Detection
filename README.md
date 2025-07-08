Project Name - COPD (Chronic Obstructive Pulmonary Disease) detection using TinyML

Project Overview - This project implements a TinyML based Artificial Neural Network (ANN) for Chronic Obstructive Pulmonary Disease (COPD) detection, running on an ESP32 development board. Using TensorFlow Lite for Microcontrollers (TFLite Micro), the trained ANN model is deployed on-device to perform real-time inference using sensors data. The system is optimized for low-power, low-latency health diagnostics in resource-constrained environments.

Input Features - 
The list of sensors used in this project:
1. MQ-3: Used to measure Alcohol concentration
2. MQ-7: Used to measure Carbon Monoxide (CO) level
3. MQ-135: Used to measure VOCs (ammonia, NOx, benzene)
4. MAX30102: Used to measure Heart Rate (BPM) and SpO2
5. GP2Y1010: Used to measure dust (mg/m3)
6. MLX90614: Used to measure body temperature (°C)

ANN model details -
Model Type: Artificial Neural Network (ANN)
Framework: TensorFlow (trained) → TFLite Micro (deployed)
Model Input: 7 normalized float values
Model Ouput: Softmax probabilities for 3 classes:
0 → At Risk
1 → COPD
2 → Healthy

Project Working -
Model Prediction is triggered by a button press (GPIO 4). As soon as the button is pressed, sensor readings are collected and processed. The model performs on-device inference and the predicted class is displayed on the Serial Monitor as well as the 16X2 I2C LCD.