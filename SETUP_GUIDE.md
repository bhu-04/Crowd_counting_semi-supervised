# 🚀 AI Crowd Counter - Complete Setup Guide

## Overview
This project combines **semi-supervised learning** with a **full-stack web application** for crowd counting. It can work with both labeled and unlabeled data!

---

## 📁 Project Structure

```
crowd-counter/
├── model.py                    # CrowdCCT architecture
├── dataset.py                  # Original dataset loader
├── enhanced_dataset.py         # NEW: Mixed labeled/unlabeled dataset
├── pseudo_labeler.py          # NEW: Generate labels for unlabeled data
├── train.py                   # Original supervised training
├── semi_supervised_train.py   # NEW: Semi-supervised training
├── eval.py                    # Evaluation script
├── utils.py                   # Helper functions
├── app.py                     # NEW: Flask backend API
├── index.html                 # NEW: Web frontend
├── requirements.txt           # Python dependencies
├── data/
│   ├── train/
│   │   ├── images/           # Labeled training images
│   │   └── annots/           # Ground truth .mat files
│   ├── test/
│   │   ├── images/           # Test images
│   │   └── annots/           # Test annotations
│   └── unlabeled/
│       └── images/           # NEW: Unlabeled images (no annotations needed!)
└── outputs/
    ├── best_model.pth        # Trained model weights
    └── pseudo_labels.json    # Generated pseudo-labels
```

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

#### Option A: You Have Labeled Data Only
```
data/
├── train/
│   ├── images/
│   └── annots/  # .mat files with crowd counts
└── test/
    ├── images/
    └── annots/
```

#### Option B: You Have Mixed Data (Labeled + Unlabeled)
```
data/
├── train/
│   ├── images/      # Your labeled images
│   └── annots/      # .mat files
├── unlabeled/
│   └── images/      # Just put images here! No annotations needed!
└── test/
    ├── images/
    └── annots/
```

---

## 🎯 Training Options

### Option 1: Traditional Supervised Training (Labeled Data Only)

```bash
python train.py
```

This uses only your labeled data with ground truth annotations.

### Option 2: Semi-Supervised Training (Labeled + Unlabeled)

```bash
python semi_supervised_train.py
```

**What happens:**
1. Trains on your labeled data
2. Generates pseudo-labels for unlabeled images using:
   - Faster R-CNN person detection
   - Your trained CrowdCCT model
   - Ensemble of both methods
3. Trains on unlabeled data with confidence-weighted loss
4. Periodically regenerates better pseudo-labels

**Benefits:**
- Works with unlimited unlabeled images
- Improves model diversity and generalization
- No manual labeling required for new data!

---

## 🌐 Running the Web Application

### Step 1: Start the Backend API

```bash
python app.py
```

This starts the Flask server on `http://localhost:5000`

**Endpoints:**
- `GET /health` - Check API status
- `POST /predict` - Single image prediction
- `POST /batch_predict` - Multiple images
- `GET /model_info` - Model architecture info

### Step 2: Open the Frontend

Simply open `index.html` in your web browser!

Or use a local server:
```bash
# Python 3
python -m http.server 8000

# Then visit: http://localhost:8000
```

### Step 3: Use the App

1. **Drag & drop** or **click to upload** an image
2. Wait for AI analysis (2-5 seconds)
3. View results:
   - Original image
   - Density heatmap overlay
   - Precise people count
   - Confidence score
4. Download report as JSON

---

## 📊 Evaluation

Test your trained model:

```bash
python eval.py
```

**Outputs:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- Visualization images in `outputs/result_*.png`

---

## 🎨 Features

### Semi-Supervised Learning
✅ Works with unlimited unlabeled images  
✅ Automatic pseudo-label generation  
✅ Ensemble prediction (Detection + Crowd Model)  
✅ Confidence-weighted training  
✅ Iterative label refinement  

### Web Application
✅ Drag-and-drop interface  
✅ Real-time predictions  
✅ Density heatmap visualization  
✅ Batch processing support  
✅ Downloadable reports  
✅ Mobile-responsive design  

### Model Architecture
✅ DenseNet-121 backbone  
✅ Multi-scale dilated attention  
✅ Location-enhanced attention  
✅ Fusion mechanism  

---

## 🔥 Quick Start Guide

**If you're starting fresh:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your labeled images in data/train/
# 3. Put ANY crowd images in data/unlabeled/ (no labels needed!)

# 4. Train with semi-supervised learning
python semi_supervised_train.py

# 5. Start the web app
python app.py

# 6. Open index.html in browser and upload images!
```

---

## 🎓 How It Works

### Pseudo-Labeling Process

1. **Faster R-CNN Detection**: Counts individual people (good for sparse crowds)
2. **Crowd Model Prediction**: Uses density estimation (good for dense crowds)
3. **Ensemble Decision**: 
   - If detected people > 20 → Trust detection (70%)
   - If detected people < 10 → Trust crowd model (70%)
   - Middle ground → 50/50 blend

4. **Confidence Weighting**: Only high-confidence pseudo-labels are used
5. **Iterative Refinement**: Labels improve as model trains

### Training Strategy

```python
Total Loss = Labeled Loss + λ × (Confidence × Unlabeled Loss)
```

- λ = 0.5 (unlabeled weight)
- Confidence ∈ [0, 1] (pseudo-label quality)
- Early stopping with patience=10

---

## 🚨 Troubleshooting

### "Model file not found"
- Train the model first using `train.py` or `semi_supervised_train.py`

### "API server not responding"
- Make sure `app.py` is running: `python app.py`
- Check if port 5000 is available

### "CUDA out of memory"
- Reduce batch size in training scripts (default: 16 → 8 or 4)
- Use smaller images or CPU: `device = torch.device("cpu")`

### "No such file or directory: data/unlabeled"
- Create the directory: `mkdir -p data/unlabeled/images`
- Or skip unlabeled training and use `train.py`

---

## 📈 Performance Tips

1. **More unlabeled data = Better generalization**
   - Add diverse crowd scenarios
   - Different lighting, angles, densities

2. **Adjust confidence threshold**
   - In `semi_supervised_train.py`: `min_confidence=0.4`
   - Higher = stricter (fewer but better labels)
   - Lower = more labels (but noisier)

3. **Fine-tune λ (unlabeled weight)**
   - Current: `lambda_u = 0.5`
   - Increase if you trust pseudo-labels
   - Decrease if labeled data is very accurate

---

## 🎉 What Makes This Different?

1. **No need to label everything!** - Just throw in crowd images
2. **Web interface included** - Not just Python scripts
3. **Production-ready API** - Easy to integrate anywhere
4. **Visual feedback** - See density heatmaps in real-time
5. **Scalable** - Works on any device with a browser

---

## 📝 Next Steps

- [ ] Deploy to cloud (AWS, Google Cloud, Heroku)
- [ ] Add video support (count crowds in videos)
- [ ] Real-time webcam counting
- [ ] Mobile app (React Native / Flutter)
- [ ] Database integration for analytics
- [ ] Multi-camera support
- [ ] Export to PDF reports

---

## 🤝 Contributing

Feel free to:
- Add more crowd datasets
- Improve pseudo-labeling algorithms
- Enhance UI/UX design
- Add new features

---

## 📄 License

MIT License - Use freely!

---

**Need help?** Open an issue or contact the maintainer!

**Happy Counting! 🎯**