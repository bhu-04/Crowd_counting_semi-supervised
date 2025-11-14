# 👥 AI Crowd Counter

<div align="center">

![Crowd Counter Banner](https://img.shields.io/badge/AI-Crowd%20Counter-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.3+-black?style=for-the-badge&logo=flask)

**Advanced Deep Learning System for Precise Crowd Counting and Analysis**

[📺 Watch Demo](https://youtu.be/ikP2KuRq6Ow) • [📁 Download Dataset](https://www.dropbox.com/scl/fo/2lu0e2hyivphcr943as3m/AF44tvTT9URXmdxoZas_kJI?rlkey=94jo375jk2lya1pbg4pgcxrr2&dl=0) • [📖 Documentation](#documentation)

</div>

---

## 🎯 Overview

AI Crowd Counter is a state-of-the-art computer vision system that combines **semi-supervised learning** with a **full-stack web application** to accurately count people in crowd images. The system uses a hybrid CNN-Transformer architecture (CrowdCCT) and can work with both labeled and unlabeled data.

### ✨ Key Features

- 🧠 **Hybrid Architecture**: CNN backbone + Transformer attention mechanisms
- 🔄 **Semi-Supervised Learning**: Leverages unlimited unlabeled images
- 🎨 **Interactive Web Interface**: Real-time predictions with heatmap visualizations
- 📊 **Patch-Based Processing**: Handles extremely large crowds (10K+ people)
- 🚀 **Production Ready**: Docker support, REST API, and deployment guides
- 📈 **Pseudo-Labeling**: Automatic label generation for unlabeled data

---

## 📺 Video Demo

**Watch the full demonstration and tutorial:**

[![AI Crowd Counter Demo](https://img.youtube.com/vi/ikP2KuRq6Ow/maxresdefault.jpg)](https://youtu.be/ikP2KuRq6Ow)

🔗 **Direct Link**: https://youtu.be/ikP2KuRq6Ow

---

## 📊 Dataset

### Download the Dataset

The project uses a curated crowd counting dataset with labeled and unlabeled images.

**📥 [Download Dataset from Dropbox](https://www.dropbox.com/scl/fo/2lu0e2hyivphcr943as3m/AF44tvTT9URXmdxoZas_kJI?rlkey=94jo375jk2lya1pbg4pgcxrr2&dl=0)**

### Dataset Structure

After downloading, organize your data as follows:

```
data/
├── train/
│   ├── images/          # Labeled training images
│   └── annots/          # Ground truth .mat files
├── test/
│   ├── images/          # Test images
│   └── annots/          # Test annotations
└── unlabeled/
    └── images/          # Unlabeled images (no annotations needed)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ disk space

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd crowd-counter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download and Setup Data

1. **Download dataset** from the Dropbox link above
2. **Extract** the downloaded archive
3. **Move** the folders to create the structure shown above

### Train the Model

#### Option 1: Supervised Training (Labeled Data Only)

```bash
python train.py
```

#### Option 2: Semi-Supervised Training (Labeled + Unlabeled)

```bash
python semi_supervised_train.py
```

This will:
- Train on your labeled data
- Generate pseudo-labels for unlabeled images
- Train on both labeled and pseudo-labeled data
- Iteratively refine predictions

### Run the Web Application

```bash
# Start the Flask backend
python app.py

# Open index.html in your browser
# Or serve it with:
python -m http.server 8000
```

Visit `http://localhost:8000` to use the web interface!

---

## 🏗️ Architecture

### CrowdCCT Model

```
Input Image (RGB)
    ↓
DenseNet-121 Backbone (CNN)
    ↓
Multi-Scale Dilated Attention
    ↓
Location-Enhanced Attention
    ↓
Fusion Mechanism
    ↓
Regression Head
    ↓
Crowd Count (Scalar)
```

### Key Components

- **CNN Backbone**: DenseNet-121 pretrained on ImageNet
- **Multi-Scale Dilated Attention**: Captures crowd patterns at different scales
- **Location-Enhanced Attention**: Incorporates spatial information
- **Fusion Mechanism**: Combines multi-scale and location features

---

## 💻 Usage

### Web Interface

1. **Upload Image**: Drag & drop or click to select
2. **Choose Method**:
   - **Standard**: Best for crowds ≤ 100 people
   - **Patch-Based**: For large crowds > 100 people
3. **View Results**: See count, heatmap, and confidence score
4. **Download Report**: Export results as JSON

### API Endpoints

#### Standard Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@crowd.jpg"
```

#### Patch-Based Prediction (Large Crowds)
```bash
curl -X POST http://localhost:5000/predict_large_crowd \
  -F "image=@large_crowd.jpg" \
  -F "patch_size=384" \
  -F "overlap=96"
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg"
```

#### Model Info
```bash
curl http://localhost:5000/model_info
```

---

## 🧪 Evaluation

Test your trained model:

```bash
python eval.py
```

**Output Metrics:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- Visualization images in `outputs/result_*.png`

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# Access services
# API: http://localhost:5000
# Web: http://localhost:8080

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t crowd-counter .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  crowd-counter
```

---

## 📁 Project Structure

```
crowd-counter/
├── model.py                    # CrowdCCT architecture
├── dataset.py                  # Dataset loader
├── enhanced_dataset.py         # Mixed labeled/unlabeled dataset
├── pseudo_labeler.py          # Pseudo-label generation
├── train.py                   # Supervised training
├── semi_supervised_train.py   # Semi-supervised training
├── eval.py                    # Evaluation script
├── utils.py                   # Helper functions
├── app.py                     # Flask API backend
├── index.html                 # Web frontend
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── SETUP_GUIDE.md            # Detailed setup instructions
├── DEPLOYMENT.md             # Deployment guide
├── data/                     # Dataset directory
│   ├── train/
│   ├── test/
│   └── unlabeled/
└── outputs/                  # Model checkpoints & results
    ├── best_model.pth
    └── pseudo_labels.json
```

---

## 🎓 How It Works

### Semi-Supervised Learning Pipeline

1. **Initial Training**: Train CrowdCCT on labeled data
2. **Pseudo-Labeling**: Generate labels for unlabeled images using:
   - Faster R-CNN person detection (for sparse crowds)
   - Trained CrowdCCT model (for dense crowds)
   - Ensemble voting for final prediction
3. **Confidence Weighting**: Only high-confidence labels are used
4. **Joint Training**: Train on both labeled and pseudo-labeled data
5. **Iterative Refinement**: Regenerate better pseudo-labels every 10 epochs

### Patch-Based Processing

For extremely large crowds (stadium events, protests, festivals):

1. Divide image into overlapping patches (384×384)
2. Process each patch independently
3. Apply Gaussian weighting to patch predictions
4. Stitch results with weighted averaging
5. Output final count and density map

---

## 📈 Performance

### Typical Results

| Metric | Value |
|--------|-------|
| MAE on Test Set | ~5-15 people |
| Inference Time | 2-5 seconds (single image) |
| Supported Crowd Size | 1 - 100,000+ people |
| Model Size | ~150MB |

### Optimization Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch Processing**: Process multiple images simultaneously
3. **Model Quantization**: Reduce size by 4x with minimal accuracy loss
4. **Cache Results**: Store predictions for repeated queries

---

## 🔧 Configuration

### Training Parameters

Edit in `semi_supervised_train.py`:

```python
TOTAL_EPOCHS = 100          # Number of training epochs
FINE_TUNE_LR = 5e-6         # Learning rate for fine-tuning
lambda_u = 0.5              # Weight for unlabeled loss
patience = 10               # Early stopping patience
min_confidence = 0.4        # Minimum pseudo-label confidence
```

### Patch Processing

Adjust in API request:

```python
patch_size = 384            # Size of each patch
overlap = 96                # Overlap between patches
```

---

## 🚀 Deployment Options

### Cloud Platforms

- **Heroku**: Free tier available ([Guide](DEPLOYMENT.md#option-2-heroku))
- **AWS EC2**: Full control, GPU support ([Guide](DEPLOYMENT.md#option-3-aws-ec2))
- **Google Cloud Run**: Serverless, auto-scaling ([Guide](DEPLOYMENT.md#option-4-google-cloud-run))
- **DigitalOcean**: Simple droplets ([Guide](DEPLOYMENT.md#option-5-digitalocean))

### Production Checklist

- [ ] Use Gunicorn/uWSGI instead of Flask dev server
- [ ] Setup Nginx reverse proxy
- [ ] Enable HTTPS with Let's Encrypt
- [ ] Add rate limiting
- [ ] Implement monitoring (Prometheus/Grafana)
- [ ] Configure backups
- [ ] Add authentication (if needed)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides.

---

## 📚 Documentation

- **[Setup Guide](SETUP_GUIDE.md)**: Complete installation and setup instructions
- **[Deployment Guide](DEPLOYMENT.md)**: Production deployment options
- **[API Documentation](#api-endpoints)**: REST API reference
- **[Video Tutorial](https://youtu.be/ikP2KuRq6Ow)**: Visual walkthrough

---

## 🛠️ Troubleshooting

### Common Issues

**Model not found error:**
```bash
# Train the model first
python train.py  # or semi_supervised_train.py
```

**API not responding:**
```bash
# Check if Flask is running
python app.py

# Verify port 5000 is available
netstat -an | grep 5000  # Unix
netstat -an | findstr 5000  # Windows
```

**CUDA out of memory:**
```python
# Reduce batch size in training scripts
# Edit train.py or semi_supervised_train.py
batch_size = 8  # or 4
```

**Pseudo-labels not generated:**
```bash
# Ensure unlabeled images directory exists
mkdir -p data/unlabeled/images

# Check if images are present
ls data/unlabeled/images
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add video processing support
- [ ] Implement real-time webcam counting
- [ ] Create mobile app (React Native/Flutter)
- [ ] Add more visualization options
- [ ] Improve pseudo-labeling algorithms
- [ ] Database integration for analytics
- [ ] Multi-camera support

---

## 📄 License

MIT License - Use freely for commercial and non-commercial projects.

---

## 🙏 Acknowledgments

- DenseNet architecture from torchvision
- Faster R-CNN for person detection
- Crowd counting research community
- Dataset contributors

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: Check [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Video Tutorial**: [Watch on YouTube](https://youtu.be/ikP2KuRq6Ow)

---

## 🎯 Roadmap

- [x] Semi-supervised learning support
- [x] Web interface with visualizations
- [x] Patch-based processing for large crowds
- [x] Docker deployment
- [ ] Video stream processing
- [ ] Real-time webcam support
- [ ] Mobile application
- [ ] Cloud-based inference API
- [ ] Multi-language support

---

## 📊 Citations

If you use this project in your research, please cite:

```bibtex
@software{ai_crowd_counter,
  title = {AI Crowd Counter: Semi-Supervised Deep Learning for Crowd Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/crowd-counter}
}
```

---

<div align="center">

**Made with ❤️ for the Computer Vision Community**

[⭐ Star this repo](.) • [🐛 Report Bug](.) • [✨ Request Feature](.)

</div>
