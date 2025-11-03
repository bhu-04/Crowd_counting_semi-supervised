from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as T
from PIL import Image
import io
import base64
import numpy as np
import cv2
from model import CrowdCCT
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor()
])

def load_model():
    """Load the trained model"""
    global model
    model = CrowdCCT().to(device)
    
    model_path = 'outputs/best_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return True
    else:
        print("Warning: Model file not found. Using untrained model.")
        model.eval()
        return False

# ============================================
# PATCH-BASED COUNTING FUNCTIONS (NEW)
# ============================================

def count_large_crowd(image, model, device, patch_size=384, overlap=96):
    """
    Count people in large crowds by dividing into overlapping patches
    """
    # Use same normalization as training
    transform_patch = T.Compose([
        T.ToTensor()
    ])
    
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    h, w = img_np.shape[:2]
    print(f"Processing image of size: {w}x{h}")
    
    density_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    stride = patch_size - overlap
    num_patches_y = max(1, (h - overlap) // stride + 1)
    num_patches_x = max(1, (w - overlap) // stride + 1)
    total_patches = num_patches_y * num_patches_x
    
    print(f"Dividing into {num_patches_y}x{num_patches_x} = {total_patches} patches")
    
    model.eval()
    patch_count = 0
    
    with torch.no_grad():
        for y in range(0, max(h - patch_size + 1, 1), stride):
            for x in range(0, max(w - patch_size + 1, 1), stride):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                y_start = max(0, y_end - patch_size)
                x_start = max(0, x_end - patch_size)
                
                patch = img_np[y_start:y_end, x_start:x_end]
                
                # Resize patch to model input size
                patch_pil = Image.fromarray(patch)
                patch_resized = patch_pil.resize((384, 384))
                patch_tensor = transform(patch_resized).unsqueeze(0).to(device)
                
                pred_count = model(patch_tensor).item()
                
                # Create simple uniform density for this patch
                patch_h = y_end - y_start
                patch_w = x_end - x_start
                patch_density = np.ones((patch_h, patch_w), dtype=np.float32) * (pred_count / (patch_h * patch_w))
                
                # Gaussian weight
                y_coords, x_coords = np.ogrid[:patch_h, :patch_w]
                center_y, center_x = patch_h / 2, patch_w / 2
                sigma = min(patch_h, patch_w) / 4
                
                gaussian_weight = np.exp(
                    -((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2)
                )
                
                density_map[y_start:y_end, x_start:x_end] += patch_density * gaussian_weight
                weight_map[y_start:y_end, x_start:x_end] += gaussian_weight
                
                patch_count += 1
                if patch_count % 10 == 0:
                    print(f"Processed {patch_count}/{total_patches} patches...")
    
    density_map = density_map / (weight_map + 1e-6)
    total_count = density_map.sum()
    
    print(f"✓ Processing complete! Estimated count: {total_count:.0f}")
    return total_count, density_map


def visualize_density_map(image, density_map):
    """Create heatmap visualization from density map"""
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    if density_map.shape != (h, w):
        density_map = cv2.resize(density_map, (w, h))
    
    # Normalize for visualization
    density_viz = (density_map / (density_map.max() + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(density_viz, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return overlay

# ============================================
# EXISTING FUNCTIONS (keep as is)
# ============================================

def generate_heatmap(image, count_prediction):
    """Generate heatmap based on image features"""
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    heatmap = cv2.GaussianBlur(edges.astype(np.float32), (51, 51), 0)
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8) if heatmap.max() > 0 else heatmap
    
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Standard prediction endpoint (existing)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_size = image.size
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)
            count = max(0, round(prediction.item()))
        
        heatmap = generate_heatmap(image, count)
        heatmap_b64 = image_to_base64(heatmap)
        original_b64 = image_to_base64(np.array(image))
        
        response = {
            'count': int(count),
            'confidence': 0.85,
            'original_image': original_b64,
            'heatmap': heatmap_b64,
            'image_size': {
                'width': original_size[0],
                'height': original_size[1]
            },
            'timestamp': datetime.now().isoformat(),
            'method': 'single_image'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_large_crowd', methods=['POST'])
def predict_large_crowd():
    """
    NEW: Patch-based prediction for large crowds
    Use this for images with expected 10K+ people
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_size = image.size
        
        # Get patch parameters from request (optional)
        patch_size = int(request.form.get('patch_size', 384))
        overlap = int(request.form.get('overlap', 96))
        
        print(f"\n{'='*50}")
        print(f"Starting patch-based counting...")
        print(f"Image size: {original_size}")
        print(f"Patch size: {patch_size}, Overlap: {overlap}")
        print(f"{'='*50}\n")
        
        # Run patch-based counting
        start_time = datetime.now()
        count, density_map = count_large_crowd(
            image, 
            model, 
            device,
            patch_size=patch_size,
            overlap=overlap
        )
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate visualization
        heatmap = visualize_density_map(image, density_map)
        heatmap_b64 = image_to_base64(heatmap)
        original_b64 = image_to_base64(np.array(image))
        
        response = {
            'count': int(count),
            'confidence': 0.75,  # Lower confidence for extrapolation
            'original_image': original_b64,
            'heatmap': heatmap_b64,
            'image_size': {
                'width': original_size[0],
                'height': original_size[1]
            },
            'processing_time': f'{processing_time:.2f}s',
            'timestamp': datetime.now().isoformat(),
            'method': 'patch_based',
            'patch_info': {
                'patch_size': patch_size,
                'overlap': overlap
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple images"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(img_tensor)
                count = max(0, round(prediction.item()))
            
            results.append({
                'filename': file.filename,
                'count': int(count),
                'confidence': 0.85
            })
        
        return jsonify({
            'results': results,
            'total_count': sum(r['count'] for r in results),
            'num_images': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    info = {
        'architecture': 'CrowdCCT (CNN + Transformer)',
        'backbone': 'DenseNet-121',
        'input_size': '384x384',
        'device': str(device),
        'parameters': sum(p.numel() for p in model.parameters()) if model else 0,
        'patch_based_available': True
    }
    return jsonify(info)

if __name__ == '__main__':
    print("Starting Crowd Counting API Server...")
    print(f"Using device: {device}")
    
    load_model()
    
    print("\nAvailable endpoints:")
    print("  - POST /predict              (standard single-image)")
    print("  - POST /predict_large_crowd  (patch-based for large crowds)")
    print("  - POST /batch_predict        (multiple images)")
    print("  - GET  /model_info           (model information)")
    print("  - GET  /health               (health check)")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)