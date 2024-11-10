# setup_project.py

import os
from pathlib import Path
import yaml

class SmartDetectionSetup:
    def __init__(self):
        self.base_dir = Path("smart_detection_system")
        self.directories = {
            "app": [
                "core",
                "detectors",
                "services",
                "utils"
            ],
            "data": [
                "weights",
                "images",
                "logs"
            ],
            "config": [],
            "models": [],
            "tests": [],
            "arduino": []
        }

        self.files = {
            "config/config.yaml": self.get_config_content(),
            "config/products.csv": self.get_products_csv_content(),
            "app/core/__init__.py": "",
            "app/core/config.py": self.get_core_config_content(),
            "app/core/database.py": self.get_database_content(),
            "app/detectors/__init__.py": "",
            "app/detectors/weight_detector.py": self.get_weight_detector_content(),
            "app/detectors/object_detector.py": self.get_object_detector_content(),
            "app/services/__init__.py": "",
            "app/services/detection_service.py": self.get_detection_service_content(),
            "app/utils/__init__.py": "",
            "app/utils/weight_utils.py": self.get_weight_utils_content(),
            "arduino/weight_sensor.ino": self.get_arduino_content(),
            "main.py": self.get_main_content(),
            "requirements.txt": self.get_requirements_content()
        }

    def create_structure(self):
        """Create the project directory structure"""
        # Create directories
        for dir_name, subdirs in self.directories.items():
            main_dir = self.base_dir / dir_name
            main_dir.mkdir(parents=True, exist_ok=True)

            for subdir in subdirs:
                (main_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create files
        for file_path, content in self.files.items():
            full_path = self.base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

    def get_core_config_content(self):
        return '''
from pydantic import BaseSettings
from typing import List, Dict
from pathlib import Path

class WeightSensorConfig(BaseSettings):
    port: str = "/dev/ttyACM0"
    baudrate: int = 9600
    timeout: float = 1.0
    weight_threshold: float = 10.0

class CameraConfig(BaseSettings):
    device_ids: List[int]
    resolution: List[int]
    fps: int = 30

class ModelConfig(BaseSettings):
    path: str
    confidence: float = 0.5
    device: str = "cuda"

class Settings(BaseSettings):
    app_name: str = "Smart Detection System"
    debug: bool = False
    weight_sensor: WeightSensorConfig
    camera: CameraConfig
    model: ModelConfig
    products_csv: Path
    database_path: Path
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
'''

    def get_database_content(self):
        return '''
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    weight = Column(Float)
    product_id = Column(Integer)
    product_name = Column(String)
    confidence = Column(Float)
    camera_id = Column(Integer)

def init_db(database_url: str):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
'''

    def get_weight_detector_content(self):
        return '''
import serial
import time
import threading
from queue import Queue
from typing import Optional, Tuple
from ..core.logger import setup_logger

class WeightDetector:
    def __init__(self, port: str, baudrate: int = 9600, threshold: float = 10.0):
        self.port = port
        self.baudrate = baudrate
        self.threshold = threshold
        self.logger = setup_logger("weight_detector")
        self.serial_conn: Optional[serial.Serial] = None
        self.weight_queue = Queue()
        self.running = False
        self.last_weight = 0.0

    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.logger.info(f"Connected to weight sensor on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to weight sensor: {e}")
            return False

    def start(self):
        if not self.serial_conn and not self.connect():
            raise RuntimeError("Cannot start without serial connection")

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_weight)
        self.monitor_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        if self.serial_conn:
            self.serial_conn.close()

    def _monitor_weight(self):
        while self.running:
            if self.serial_conn.in_waiting:
                try:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    weight = float(line)

                    if abs(weight - self.last_weight) >= self.threshold:
                        self.weight_queue.put((weight, time.time()))
                        self.logger.debug(f"Weight change detected: {weight}g")

                    self.last_weight = weight
                except ValueError:
                    self.logger.warning(f"Invalid weight reading: {line}")
                except Exception as e:
                    self.logger.error(f"Error reading weight: {e}")

            time.sleep(0.01)
'''

    def get_object_detector_content(self):
        return '''
import cv2
import torch
import numpy as np
from typing import List, Dict
from ..core.logger import setup_logger

class ObjectDetector:
    def __init__(self, model_path: str, confidence: float = 0.5):
        self.logger = setup_logger("object_detector")
        self.confidence = confidence
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._load_model(model_path)
        self.logger.info(f"Model loaded successfully on {self.device}")

    def _load_model(self, model_path: str):
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.to(self.device)
            model.conf = self.confidence
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame)
        detections = []

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(float, det)
            if conf >= self.confidence:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': results.names[int(cls)],
                    'confidence': conf
                })

        return detections
'''

    def get_detection_service_content(self):
        return '''
import cv2
import time
import pandas as pd
from typing import Dict, List, Optional
from queue import Empty
from ..detectors.weight_detector import WeightDetector
from ..detectors.object_detector import ObjectDetector
from ..core.logger import setup_logger
from pathlib import Path

class DetectionService:
    def __init__(self, config: Dict):
        self.logger = setup_logger("detection_service")
        self.config = config
        self.running = False

        # Initialize detectors
        self.weight_detector = WeightDetector(
            port=config['weight_sensor']['port'],
            threshold=config['weight_sensor']['weight_threshold']
        )

        self.object_detector = ObjectDetector(
            model_path=config['model']['path'],
            confidence=config['model']['confidence']
        )

        # Load product database
        self.products_df = pd.read_csv(config['products_csv'])

        # Initialize cameras
        self.cameras = self._init_cameras()

    def _init_cameras(self):
        cameras = {}
        for device_id in self.config['camera']['device_ids']:
            cap = cv2.VideoCapture(device_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution'][1])
            cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            cameras[device_id] = cap
        return cameras

    def start(self):
        self.weight_detector.start()
        self.running = True

        try:
            while self.running:
                try:
                    weight, timestamp = self.weight_detector.weight_queue.get(timeout=0.1)
                    self.handle_detection(weight, timestamp)
                except Empty:
                    continue
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.weight_detector.stop()
        for cap in self.cameras.values():
            cap.release()
        cv2.destroyAllWindows()

    def handle_detection(self, weight: float, timestamp: float):
        # Get possible products based on weight
        possible_products = self._get_products_by_weight(weight)

        if possible_products:
            # Process all cameras
            detections = self._process_cameras()

            # Match products
            matched_product = self._match_product(possible_products, detections)

            if matched_product:
                self.logger.info(
                    f"Product detected: {matched_product['name']} "
                    f"(confidence: {matched_product['confidence']:.2f})"
                )

    def _get_products_by_weight(self, weight: float, tolerance: float = 1.0):
        return self.products_df[
            (self.products_df['min_weight'] - tolerance <= weight) &
            (self.products_df['max_weight'] + tolerance >= weight)
        ].to_dict('records')

    def _process_cameras(self):
        detections = {}
        for camera_id, cap in self.cameras.items():
            ret, frame = cap.read()
            if ret:
                detections[camera_id] = self.object_detector.detect(frame)
        return detections

    def _match_product(self, possible_products, detections):
        # Implementation of product matching logic
        pass
'''

    def get_weight_utils_content(self):
        return '''
def calculate_moving_average(readings: list, window_size: int = 5) -> float:
    """Calculate moving average of weight readings"""
    if not readings:
        return 0.0
    window = readings[-window_size:]
    return sum(window) / len(window)

def is_weight_stable(readings: list, threshold: float = 0.5) -> bool:
    """Check if weight readings are stable"""
    if len(readings) < 3:
        return False
    return all(abs(readings[i] - readings[i-1]) < threshold for i in range(1, len(readings)))
'''

    def get_arduino_content(self):
        return '''
#include "HX711.h"

// Pin definitions
const int LOADCELL_DOUT_PIN = 2;
const int LOADCELL_SCK_PIN = 3;
const int TARE_BUTTON_PIN = 4;
const int STATUS_LED_PIN = 13;

// HX711 configuration
HX711 scale;
float calibration_factor = -96650.0;
float last_stable_weight = 0.0;
const float STABILITY_THRESHOLD = 0.5;

// Timing
unsigned long last_send_time = 0;
const unsigned long SEND_INTERVAL = 100;

void setup() {
    Serial.begin(9600);
    pinMode(TARE_BUTTON_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);

    scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
    scale.set_scale(calibration_factor);
    scale.tare();

    digitalWrite(STATUS_LED_PIN, HIGH);
    delay(1000);
    digitalWrite(STATUS_LED_PIN, LOW);
}

void loop() {
    if (digitalRead(TARE_BUTTON_PIN) == LOW) {
        scale.tare();
        digitalWrite(STATUS_LED_PIN, HIGH);
        delay(500);
        digitalWrite(STATUS_LED_PIN, LOW);
    }

    if (millis() - last_send_time >= SEND_INTERVAL) {
        float weight = scale.get_units(3);
        Serial.println(weight, 2);
        last_send_time = millis();
    }
}
'''

    def get_main_content(self):
        return '''
import yaml
import argparse
from pathlib import Path
from app.services.detection_service import DetectionService
from app.core.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Smart Detection System")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.debug:
        config['debug'] = True

    # Initialize and start service
    service = DetectionService(config)
    service.start()

if __name__ == "__main__":
    main()
'''

    @staticmethod
    def get_config_content():
        config = {
            "app_name": "Smart Detection System",
            "debug": True,
            "weight_sensor": {
                "port": "/dev/ttyACM0",
                "baudrate": 9600,
                "weight_threshold": 10.0
            },
            "camera": {
                "device_ids": [0, 1, 2],
                "resolution": [1920, 1080],
                "fps": 30
            },
            "model": {
                "path": "models/yolov5s.pt",
                "confidence": 0.5
            },
            "database": {
                "path": "data/detections.db"
            }
        }
        return yaml.dump(config, default_flow_style=False)

    @staticmethod
    def get_products_csv_content():
        return """product_id,name,min_weight,max_weight,class_name,description
1,Coca Cola Can,330,335,CokeCan330,Standard 330ml Coca Cola Can
2,Pepsi Can,330,335,PepsiCan330,Standard 330ml Pepsi Can
3,Lays Classic 40g,38,42,LaysClassic40,Lays Classic Potato Chips 40g
4,Doritos 40g,38,42,Doritos40,Doritos Nacho Cheese 40g
5,Snickers 50g,48,52,Snickers50,Snickers Chocolate Bar 50g
6,Mountain Dew 500ml,500,505,MtnDew500,Mountain Dew 500ml Bottle
7,Pringles Original,165,170,PringlesOrig,Pringles Original 165g Can
8,KitKat 45g,44,46,KitKat45,KitKat 4 Finger Bar 45g
9,Mars Bar 51g,50,52,MarsBar51,Mars Chocolate Bar 51g
10,Red Bull 250ml,250,255,RedBull250,Red Bull Energy Drink 250ml"""

    @staticmethod
    def get_requirements_content():
        return """numpy>=1.19.5
opencv-python>=4.5.3
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
pyserial>=3.5
pyyaml>=5.4.1
pandas>=1.3.0
sqlalchemy>=1.4.23
pydantic>=1.8.2
pytest>=6.2.5
pillow>=8.2.0
scikit-image>=0.18.1
python-dotenv>=0.19.0
fastapi>=0.68.0
uvicorn>=0.15.0
"""

    def create_readme(self):
        readme_content = """# Smart Detection System

An integrated system for product detection combining weight sensors and computer vision.

## Features

- Real-time weight monitoring using load cell
- Multi-camera object detection using YOLOv5
- Product matching using weight and visual data
- Automatic data logging and visualization
- Weight-triggered detection system

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd smart_detection_system
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Upload Arduino code:
   - Open `arduino/weight_sensor.ino` in Arduino IDE
   - Upload to Arduino board with HX711 module

## Configuration

1. Edit `config/config.yaml` to set:
   - Serial port for weight sensor
   - Camera IDs and settings
   - Model paths and thresholds

2. Update `config/products.csv` with your product database

## Usage

1. Start the system:
   ```bash
   python main.py
   ```

2. Optional arguments:
   ```bash
   python main.py --config custom_config.yaml --debug
   ```

## Hardware Requirements

- Arduino Uno/Nano
- HX711 Load Cell Amplifier
- Load Cell Sensor
- USB Cameras (up to 3)
- NVIDIA Jetson or GPU-enabled computer

## File Structure

```
smart_detection_system/
├── app/
│   ├── core/           # Core functionality
│   ├── detectors/      # Detection modules
│   ├── services/       # Integration services
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── data/               # Data storage
├── models/             # ML models
└── arduino/            # Arduino code
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Submit pull request

## License

MIT License - see LICENSE file for details
"""
        with open(self.base_dir / "README.md", "w") as f:
            f.write(readme_content)

    def create_gitignore(self):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/debug_images/
data/*.db
models/*.pt
logs/*.log
.env

# System
.DS_Store
Thumbs.db
"""
        with open(self.base_dir / ".gitignore", "w") as f:
            f.write(gitignore_content)

    def setup_project(self):
        """Complete project setup"""
        try:
            # Create basic structure
            self.create_structure()

            # Create additional files
            self.create_readme()
            self.create_gitignore()

            # Create empty __init__.py files in all Python directories
            for root, dirs, _ in os.walk(self.base_dir):
                if "app" in root or "tests" in root:
                    for dir in dirs:
                        init_file = Path(root) / dir / "__init__.py"
                        init_file.parent.mkdir(parents=True, exist_ok=True)
                        init_file.touch()

            print("""
Smart Detection System setup completed successfully!

Next steps:
1. Create virtual environment: python -m venv venv
2. Activate virtual environment:
   - Windows: venv\\Scripts\\activate
   - Linux/Mac: source venv/bin/activate
3. Install requirements: pip install -r requirements.txt
4. Upload Arduino code to your board
5. Configure settings in config/config.yaml
6. Run the system: python main.py

For more information, see README.md
""")

        except Exception as e:
            print(f"Error during setup: {e}")
            raise

if __name__ == "__main__":
    setup = SmartDetectionSetup()
    setup.setup_project()