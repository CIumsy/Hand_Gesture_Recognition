# Hand Gesture Prediction using EMG Signals

## Overview
This project implements a **real-time hand gesture recognition model** using **3-channel EMG signals**. The model predicts **four distinct gestures** and maps them to keyboard controls for interactive applications such as gaming.

### **Recognized Gestures & Mapped Controls**
- **Bend Hand Left** → `Left Arrow (←)`
- **Bend Hand Right** → `Right Arrow (→)`
- **Flex Fingers** → `Up Arrow (↑)`
- **Pinch** → `Down Arrow (↓)`

### **Device & Communication**
These scripts are designed to work with a **custom NPG (NeuroPlayground) device** that streams real-time EMG data via **WebSocket**.

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/CIumsy/Hand_Gesture_Prediction.git
```

### **2. Install Dependencies**
Make sure you have Python installed. Then, navigate to the project directory in the command prompt and install the required libraries:
```bash
pip install -r requirements.txt
```

## Running the Model
### **1. Start Gesture Prediction**
Run the `predict.py` script to process real-time EMG signals and predict gestures:
```bash
python predict.py
```

### **2. Enable Keyboard Controls**
Once predictions are running, execute `game_controller.py` to send keypresses based on detected gestures:
```bash
python game_controller.py
```

## Usage
- Ensure that your **NPG (NeuroPlayground) device** is connected and streaming real-time data via **WebSocket**.
- Run `predict.py` first to generate predictions.
- Launch `game_controller.py` to map gestures to keyboard inputs.
- Use your hand movements to control applications that accept keyboard input.

## Future Enhancements
- **Expand Gesture Set**: Add more complex gestures for enhanced control.
- **Improve Model Accuracy**: Fine-tune the model with additional data.
- **Deploy on Embedded Devices**: Optimize for low-power microcontrollers.

## License
This project is open-source and available under the [MIT License](LICENSE).
