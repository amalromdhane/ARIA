# 🤖 Reception Robot - Complete System

A modular, state-based emotional robot system with multi-node architecture. Perfect for simulations and ready for hardware integration.

## 📋 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware Integration](#hardware-integration)
- [Node Documentation](#node-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Core Features
- **10 Robot States**: IDLE, GREETING, LISTENING, THINKING, HELPING, FAREWELL, ERROR, BUSY, SURPRISED, ANGRY
- **9 Unique Emotions**: Each with custom facial expressions
- **Multi-Node Architecture**: Modular design with separate nodes for different functions
- **Real-time Communication**: Queue-based inter-node messaging
- **Hardware Ready**: Easy integration points for sensors, cameras, servos, LEDs
- **Comprehensive Logging**: Full system event tracking
- **Control Interface**: Interactive testing and simulation

### Technical Features
- Thread-safe queue communication
- Configurable state timeouts
- State transition history
- Event logging system
- Modular node design
- Hardware abstraction layer

---

## 🏗️ Architecture

```
reception_robot_project/
├── main.py                          # Main application coordinator
├── config/
│   └── config.py                    # System configuration
├── nodes/
│   ├── gui_node.py                  # Visual display interface
│   ├── state_manager_node.py       # State management & transitions
│   ├── control_interface_node.py   # User control panel
│   ├── sensor_node.py              # Sensor input handling
│   └── logger_node.py              # System logging
├── utils/
│   └── hardware_template.py        # Hardware integration guide
├── logs/
│   └── robot.log                   # System logs (auto-created)
├── docs/
│   └── COMPLETE_GUIDE.md          # This file
├── requirements.txt                # Python dependencies
└── README.md                       # Quick reference
```

### Node Communication Flow
```
┌─────────────────┐
│  Sensor Node    │──┐
└─────────────────┘  │
                     │ Commands
┌─────────────────┐  │
│ Control Interface│──┤
└─────────────────┘  │
                     ↓
                ┌────────────────┐
                │ State Manager  │
                │    (Core)      │
                └────────────────┘
                     │
                     │ State/Emotion Updates
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
┌──────────────┐         ┌──────────────┐
│   GUI Node   │         │ Logger Node  │
└──────────────┘         └──────────────┘
```

---

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)
```bash
# Navigate to project directory
cd reception_robot_project

# Run the system (no installation needed for simulation!)
python main.py
```

This starts:
- Main GUI window (robot face display)
- Control Interface (state control buttons)
- All background nodes (state manager, sensors, logger)

### Option 2: Run Individual Nodes
```bash
# Test GUI only
python -c "from nodes.gui_node import GUINode; import queue; g = GUINode(queue.Queue(), queue.Queue()); g.run()"

# Test Control Interface only
python -c "from nodes.control_interface_node import ControlInterfaceNode; import queue; c = ControlInterfaceNode(queue.Queue()); c.run()"
```

---

## 📦 Installation

### Simulation Mode (No Hardware)
**NO INSTALLATION REQUIRED!** Just Python 3.7+

```bash
# Verify Python version
python --version  # Should be 3.7 or higher

# Run directly
python main.py
```

### Hardware Mode (Real Sensors/Camera)

#### On Ubuntu/Debian:
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y python3-opencv python3-tk

# Install Python packages
pip install opencv-python numpy --break-system-packages
```

#### On Raspberry Pi:
```bash
# Install system packages
sudo apt-get update
sudo apt-get install -y python3-opencv python3-rpi.gpio python3-tk

# Install Python packages
pip install opencv-python numpy RPi.GPIO --break-system-packages

# Enable camera
sudo raspi-config
# Navigate: Interface Options -> Camera -> Enable -> Reboot
```

#### Verify Installation:
```bash
python -c "import tkinter; print('✓ Tkinter OK')"
python -c "import cv2; print('✓ OpenCV OK')"
python -c "import numpy; print('✓ NumPy OK')"
```

---

## 📖 Usage

### Basic Operation

1. **Start the System**
   ```bash
   python main.py
   ```

2. **You'll see two windows:**
   - **GUI Window**: Shows the robot's face with emotions
   - **Control Interface**: Buttons to change states

3. **Change States**
   - Click any button in the Control Interface
   - Watch the robot's emotion change in the GUI

4. **Monitor Logs**
   - Check console for real-time events
   - View `logs/robot.log` for full history

5. **Shutdown**
   - Close either window to stop the system

### Command Line Arguments (Future)
```bash
# Run in debug mode
python main.py --debug

# Run without GUI (headless)
python main.py --no-gui

# Specify config file
python main.py --config custom_config.py
```

---

## 🔧 Hardware Integration

### Step 1: Connect Hardware

#### Ultrasonic Sensor (HC-SR04)
```
Sensor Pin -> Raspberry Pi Pin
VCC        -> 5V (Pin 2)
GND        -> GND (Pin 6)
TRIG       -> GPIO 23 (Pin 16)
ECHO       -> GPIO 24 (Pin 18) *with voltage divider!
```

⚠️ **IMPORTANT**: Use a voltage divider (1kΩ + 2kΩ resistors) on ECHO pin!

#### Camera
```bash
# USB Camera: Just plug in
# Pi Camera: Connect to camera port, enable in raspi-config
```

#### Servo Motors (SG90)
```
Servo Wire -> Pi Pin
Red (5V)   -> 5V
Brown (GND)-> GND
Orange     -> GPIO 17 (head pan)
Orange     -> GPIO 27 (head tilt)
```

### Step 2: Modify Configuration

Edit `config/config.py`:
```python
# Change simulation mode to False
ROBOT_CONFIG = {
    'simulation_mode': False,  # Enable hardware
}

# Enable sensors
SENSOR_CONFIG = {
    'ultrasonic': {
        'enabled': True,  # Enable ultrasonic
        'trigger_distance': 50,
    },
    'camera': {
        'enabled': True,  # Enable camera
    }
}
```

### Step 3: Update Sensor Node

Edit `nodes/sensor_node.py`:

Replace the `simulate_sensors()` method with:
```python
def read_real_sensors(self):
    import RPi.GPIO as GPIO
    import cv2
    
    # Setup GPIO
    GPIO.setmode(GPIO.BCM)
    TRIG = 23
    ECHO = 24
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    
    while self.running:
        # Read distance
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        
        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()
        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()
        
        distance = (pulse_end - pulse_start) * 17150
        
        # Trigger greeting if close
        if distance < 50:
            self.command_queue.put({
                'type': 'CHANGE_STATE',
                'state': 'GREETING'
            })
        
        time.sleep(0.5)
```

See `utils/hardware_template.py` for complete examples!

---

## 📚 Node Documentation

### 1. GUI Node (`gui_node.py`)
**Purpose**: Display robot face with emotions

**Responsibilities**:
- Draw robot face on canvas
- Update emotions based on state
- Provide visual feedback

**Communication**:
- **Receives**: State updates, emotion updates
- **Sends**: Nothing (display only)

**Key Methods**:
- `update_emotion(emotion)`: Change facial expression
- `draw_*_face()`: Draw specific emotions

### 2. State Manager Node (`state_manager_node.py`)
**Purpose**: Manage robot states and transitions

**Responsibilities**:
- Track current state
- Handle state transitions
- Map states to emotions
- Maintain state history

**Communication**:
- **Receives**: State change commands
- **Sends**: State updates, emotion updates

**Key Methods**:
- `change_state(new_state)`: Transition to new state
- `get_state_history()`: Retrieve transition history

### 3. Control Interface Node (`control_interface_node.py`)
**Purpose**: Provide user controls for testing

**Responsibilities**:
- Display control buttons
- Send state change commands
- System controls

**Communication**:
- **Receives**: Nothing
- **Sends**: Commands to state manager

**Key Methods**:
- `send_state_command(state)`: Request state change

### 4. Sensor Node (`sensor_node.py`)
**Purpose**: Handle sensor inputs (simulated or real)

**Responsibilities**:
- Read sensor data
- Process inputs
- Trigger automatic state changes

**Communication**:
- **Receives**: Nothing
- **Sends**: Commands based on sensor input

**Key Methods**:
- `simulate_sensors()`: Simulate sensor readings
- `read_distance_sensor()`: Read ultrasonic sensor
- `detect_face()`: Detect faces in camera

### 5. Logger Node (`logger_node.py`)
**Purpose**: Log all system events

**Responsibilities**:
- Record state changes
- Write to log file
- Console output

**Communication**:
- **Receives**: State updates (copy)
- **Sends**: Nothing

**Key Methods**:
- `log(message, level)`: Write log entry

---

## ⚙️ Configuration

### Robot Behavior
Edit `config/config.py`:

```python
# Add new states
STATES = {
    'YOUR_STATE': {
        'emotion': 'HAPPY',
        'description': 'Description',
        'timeout': 5  # Auto-return to IDLE
    }
}

# Modify sensor thresholds
SENSOR_CONFIG = {
    'ultrasonic': {
        'trigger_distance': 50,  # cm
    }
}

# Change GUI appearance
GUI_CONFIG = {
    'face_color': '#FF0000',  # Red face
    'background_color': '#000000'  # Black background
}
```

### Adding New Emotions

1. **Add to State Manager**:
   ```python
   self.states = {
       'NEW_STATE': 'NEW_EMOTION'
   }
   ```

2. **Add Drawing Method to GUI**:
   ```python
   def draw_new_emotion_face(self, x, y):
       # Draw eyes
       self.canvas.create_oval(...)
       # Draw mouth
       self.canvas.create_arc(...)
   ```

3. **Update emotion switch**:
   ```python
   elif emotion == 'NEW_EMOTION':
       self.draw_new_emotion_face(face_x, face_y)
   ```

---

## 🔧 Troubleshooting

### Problem: "No module named 'tkinter'"
**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows: Reinstall Python with tkinter checked

# macOS: Usually included, but if missing:
brew install python-tk
```

### Problem: Windows don't appear
**Solution**:
- Check if running on a system with display (not SSH without X11)
- Try: `export DISPLAY=:0` (Linux)
- Ensure not running as background process

### Problem: Sensor readings are wrong
**Solution**:
- Check GPIO pin connections
- Verify voltage divider on ultrasonic ECHO pin
- Test hardware independently first
- Check `config.py` pin numbers match wiring

### Problem: Camera not detected
**Solution**:
```bash
# Check camera connection
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Raspberry Pi: Enable camera
sudo raspi-config
```

### Problem: Permission denied on GPIO
**Solution**:
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Or run with sudo (not recommended)
sudo python main.py
```

---

## 🧪 Testing

### Test Individual Components

```bash
# Test state manager
python -c "
from nodes.state_manager_node import StateManagerNode
import queue
sm = StateManagerNode(queue.Queue(), queue.Queue(), queue.Queue())
sm.change_state('GREETING')
"

# Test GUI drawing
python -c "
from nodes.gui_node import GUINode
import queue
gui = GUINode(queue.Queue(), queue.Queue())
gui.update_emotion('HAPPY')
gui.root.mainloop()
"
```

---

## 📊 System Commands

### During Runtime

Console commands (type in terminal running main.py):
- `Ctrl+C`: Graceful shutdown
- Close any window: Stop system

### Command Queue Format
```python
# Change state
{
    'type': 'CHANGE_STATE',
    'state': 'GREETING'
}

# Get current state
{
    'type': 'GET_STATE'
}

# Shutdown
{
    'type': 'SHUTDOWN'
}
```

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Voice recognition
- [ ] Text-to-speech responses
- [ ] Web interface for remote control
- [ ] Mobile app integration
- [ ] Multiple emotion intensities
- [ ] Learning from interactions
- [ ] Multi-robot coordination

### Adding Features

1. Create new node in `nodes/`
2. Add configuration in `config/config.py`
3. Import and initialize in `main.py`
4. Add communication queues as needed

---

## 📝 License

Free for educational and commercial use.

## 👥 Contributing

To add features:
1. Create new branch
2. Add node in `nodes/`
3. Update `config.py`
4. Test thoroughly
5. Submit pull request

---

## 📞 Support

For issues:
1. Check logs in `logs/robot.log`
2. Review troubleshooting section
3. Test individual nodes
4. Check hardware connections

---

## 🎯 Examples

### Example 1: Basic Simulation
```bash
python main.py
# Click buttons in Control Interface
# Watch emotions change in GUI
```

### Example 2: Timed State Changes
```python
# Add to main.py
import time

def auto_demo():
    states = ['GREETING', 'LISTENING', 'THINKING', 'HELPING', 'FAREWELL']
    for state in states:
        command_queue.put({'type': 'CHANGE_STATE', 'state': state})
        time.sleep(3)

# Start in separate thread
threading.Thread(target=auto_demo, daemon=True).start()
```

### Example 3: Hardware Trigger
```python
# In sensor_node.py
if distance < 30:  # Very close
    self.command_queue.put({'type': 'CHANGE_STATE', 'state': 'SURPRISED'})
elif distance < 50:  # Close
    self.command_queue.put({'type': 'CHANGE_STATE', 'state': 'GREETING'})
```

---

**Made with ❤️ for robotics education and experimentation**
