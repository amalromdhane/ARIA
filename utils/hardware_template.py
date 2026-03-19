"""
Hardware Integration Template
Use this as a starting point for connecting real hardware
"""

# ==================== ULTRASONIC SENSOR ====================
"""
Hardware: HC-SR04 Ultrasonic Distance Sensor
Connections:
  - VCC -> 5V
  - GND -> GND
  - TRIG -> GPIO 23
  - ECHO -> GPIO 24 (with voltage divider!)

Code:
"""
def setup_ultrasonic():
    import RPi.GPIO as GPIO
    TRIG = 23
    ECHO = 24
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    
    return TRIG, ECHO

def read_distance(TRIG, ECHO):
    import RPi.GPIO as GPIO
    import time
    
    # Send trigger pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    # Wait for echo
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound / 2
    distance = round(distance, 2)
    
    return distance


# ==================== CAMERA & FACE DETECTION ====================
"""
Hardware: USB Webcam or Raspberry Pi Camera Module
Libraries: opencv-python, opencv-contrib-python

Code:
"""
def setup_camera():
    import cv2
    cap = cv2.VideoCapture(0)  # 0 for default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def detect_faces(cap):
    import cv2
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    ret, frame = cap.read()
    if not ret:
        return False, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return len(faces) > 0, len(faces)


# ==================== SERVO MOTORS ====================
"""
Hardware: SG90 Servo Motors
Connections:
  - Red wire -> 5V
  - Brown wire -> GND
  - Orange wire -> GPIO pin

Code:
"""
def setup_servos():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    
    servos = {
        'head_pan': 17,
        'head_tilt': 27,
    }
    
    pwm_objects = {}
    for name, pin in servos.items():
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)  # 50 Hz
        pwm.start(0)
        pwm_objects[name] = pwm
    
    return pwm_objects

def set_servo_angle(pwm, angle):
    """
    Set servo to angle (0-180 degrees)
    """
    duty = angle / 18 + 2
    pwm.ChangeDutyCycle(duty)


# ==================== LED INDICATORS ====================
"""
Hardware: LEDs with 220Ω resistors
Connections:
  - LED anode (long leg) -> Resistor -> GPIO pin
  - LED cathode (short leg) -> GND

Code:
"""
def setup_leds():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    
    led_pins = [18, 5, 6, 13]
    for pin in led_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    return led_pins

def set_led(pin, state):
    import RPi.GPIO as GPIO
    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


# ==================== SPEAKER/BUZZER ====================
"""
Hardware: Passive Buzzer or USB Speaker
For buzzer:
  - Positive -> GPIO 18
  - Negative -> GND

Code:
"""
def setup_buzzer():
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    BUZZER = 18
    GPIO.setup(BUZZER, GPIO.OUT)
    pwm = GPIO.PWM(BUZZER, 1000)  # 1000 Hz
    return pwm

def play_tone(pwm, frequency, duration):
    import time
    pwm.ChangeFrequency(frequency)
    pwm.start(50)  # 50% duty cycle
    time.sleep(duration)
    pwm.stop()


# ==================== INTEGRATION EXAMPLE ====================
"""
Complete example of integrating hardware into sensor_node.py:
"""
def sensor_node_hardware_example():
    """
    Replace the simulate_sensors() method in sensor_node.py with this
    """
    import time
    import RPi.GPIO as GPIO
    
    # Setup hardware
    TRIG, ECHO = setup_ultrasonic()
    camera = setup_camera()
    
    while self.running:
        # Read distance
        distance = read_distance(TRIG, ECHO)
        
        # Check for faces
        face_detected, num_faces = detect_faces(camera)
        
        # Trigger states based on sensors
        if face_detected and distance < 50:
            self.command_queue.put({
                'type': 'CHANGE_STATE',
                'state': 'GREETING'
            })
        
        time.sleep(0.5)  # Check twice per second
    
    # Cleanup
    GPIO.cleanup()
    camera.release()


# ==================== COMPLETE HARDWARE NODE ====================
"""
Save this as nodes/hardware_node.py and import it in main.py
"""

class HardwareNode:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.running = True
        
        # Initialize hardware
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            
            # Setup components
            self.TRIG, self.ECHO = setup_ultrasonic()
            self.camera = setup_camera()
            self.servos = setup_servos()
            self.leds = setup_leds()
            
            print("[HARDWARE_NODE] All hardware initialized")
        except Exception as e:
            print(f"[HARDWARE_NODE] ERROR: {e}")
            self.running = False
    
    def run(self):
        import time
        import RPi.GPIO as GPIO
        
        while self.running:
            try:
                # Read sensors
                distance = read_distance(self.TRIG, self.ECHO)
                face_detected, num_faces = detect_faces(self.camera)
                
                # Control logic
                if face_detected and distance < 50:
                    self.command_queue.put({
                        'type': 'CHANGE_STATE',
                        'state': 'GREETING'
                    })
                    set_led(self.leds[0], True)
                else:
                    set_led(self.leds[0], False)
                
                time.sleep(0.1)
            
            except Exception as e:
                print(f"[HARDWARE_NODE] ERROR: {e}")
        
        # Cleanup
        GPIO.cleanup()
        self.camera.release()


# ==================== INSTALLATION COMMANDS ====================
"""
Install required packages for hardware:

sudo apt-get update
sudo apt-get install -y python3-opencv python3-rpi.gpio
pip3 install opencv-python opencv-contrib-python --break-system-packages
pip3 install RPi.GPIO --break-system-packages

Enable camera (Raspberry Pi):
sudo raspi-config
  -> Interface Options -> Camera -> Enable
"""
