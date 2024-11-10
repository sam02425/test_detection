# Test Detection2: Multi-Modal Product Detection & Classification System

A comprehensive system for detecting and classifying products in images using multiple approaches including object detection (YOLO), OCR (TrOCR & PaddleOCR), and advanced product matching techniques.

## 🌟 Features

- **Multi-Modal Detection**
  - YOLO-based object detection
  - Dual OCR processing (TrOCR & PaddleOCR)
  - Vector-based similarity matching

- **Advanced Product Matching**
  - Fuzzy string matching
  - Vector database integration (FAISS)
  - SQL-based matching with keywords

- **Performance Optimizations**
  - Image result caching
  - Asynchronous processing
  - Batch operation support

- **Extensive Configuration**
  - Multiple operating modes
  - Configurable thresholds
  - Comprehensive logging

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- SQLite3

### Required Python Packages

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- ultralytics (YOLO)
- PaddleOCR
- transformers
- sentence-transformers
- faiss-gpu (or faiss-cpu)
- opencv-python
- FastAPI
- SQLite3

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/test_detection2.git
cd test_detection2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up the database**
```bash
sqlite3 data/product_database.db < scripts/schema.sql
```

4. **Configure the system**
Edit `config.yaml` to match your requirements:
```yaml
yolo_model: "models/best.pt"
trocr_model_path: "models/trocr_model.pth"
use_vector_db: true
db_path: "data/product_database.db"
```

5. **Run the system**

Using CLI:
```bash
python -m src.product_classifier --image path/to/image.jpg --db data/product_database.db --model models/trocr_model.pth
```

Using API:
```bash
uvicorn src.product_classifier:app --host 0.0.0.0 --port 8000
```

## 💻 Usage Examples

### Python API

```python
from src.product_classifier import ProductClassifier

# Initialize classifier
classifier = ProductClassifier(
    db_path="data/product_database.db",
    trocr_model_path="models/trocr_model.pth",
    use_vector_db=True
)

# Process single image
result = classifier.process_image("path/to/image.jpg")
print(f"Detected product: {result['product']}")
print(f"Confidence: {result['confidence']}")

# Batch processing
import asyncio
results = asyncio.run(classifier.batch_process_images([
    "image1.jpg",
    "image2.jpg"
]))
```

### REST API

Classify an image:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/classify
```

Update database:
```bash
curl -X POST "http://localhost:8000/update_db?brand=TestBrand&flavor=TestFlavor&keywords=keyword1,keyword2"
```

## 📁 Project Structure

```
test_detection2/
├── src/
│   ├── groundtruth_classname.py
│   ├── evaluation_metrics.py
│   ├── object_detector.py
│   ├── product_matcher.py
│   ├── unified_product_matcher.py
│   └── product_classifier.py
├── data/
├── logs/
├── models/
├── results/
├── scripts/
├── test/
├── utils/
└── config.yaml
```

## 🔧 Configuration

The system can be configured through `config.yaml`. Key configuration options:

```yaml
# Model configurations
yolo_model: "models/best.pt"
trocr_model_path: "models/trocr_model.pth"
ppocr_model_path: "models/ppocr_model"

# Processing thresholds
ocr_threshold: 0.3
product_match_threshold: 0.5
similarity_threshold: 0.7

# Feature toggles
use_ocr: true
use_bert: true
use_vector_db: true

# Operation modes
yolo_only:
  use_yolo: true
  use_trocr: false
  use_ppocr: false

ocr_bert_vector:
  use_yolo: false
  use_trocr: true
  use_ppocr: true
```

## 🧪 Testing

Run the test suite:
```bash
python -m unittest discover test
```

## 📊 Evaluation Metrics

The system provides comprehensive evaluation metrics:
- Precision
- Recall
- F1 Score
- Detection accuracy
- OCR accuracy
- Processing time

View metrics:
```python
from src.evaluation_metrics import calculate_metrics

metrics = calculate_metrics(results)
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1_score']:.2f}")
```

## 🔄 Database Management

The system uses SQLite for product data management. Schema includes:
- Brands
- Flavors
- Brand keywords
- Flavor keywords

Update database:
```python
classifier.update_database(
    brand="Brand Name",
    flavor="Flavor Name",
    keywords=["keyword1", "keyword2"]
)
```

## 📝 Logging

Logs are stored in the `logs/` directory. Configure logging level in `config.yaml`:
```yaml
log_level: INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Sam (@sam02425)

## 🙏 Acknowledgments

- YOLO by Ultralytics
- TrOCR by Microsoft
- PaddleOCR by Baidu
- FAISS by Facebook Research
