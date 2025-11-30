import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from skimage.feature import hog

app = Flask(__name__)

# --- KONFIGURASI ---
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

# Load Model & Scaler saat Server Start
print("Loading resources...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model & Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading files: {e}")

def preprocess_image(image_bytes):
    """
    Pipeline Preprocessing:
    1. Decode Image -> 2. Resize (256x256) -> 3. Grayscale
    4. HOG Feature Extraction -> 5. Standard Scaling
    """
    # 1. Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Resize (Harus sama dengan training!)
    img_resized = cv2.resize(img, (256, 256))
    
    # 3. Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 4. HOG (Parameter harus sama persis dengan JS12_P4)
    hog_feat = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8,8),
                   cells_per_block=(2,2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    
    # 5. Scaling
    hog_feat_reshaped = hog_feat.reshape(1, -1)
    features_scaled = scaler.transform(hog_feat_reshaped)
    
    return features_scaled

@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
    <div style="text-align:center; padding:50px;">
        <h1>🌤️ Day vs Night AI Classifier 🌑</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br><br>
            <button type="submit">Cek Gambar</button>
        </form>
    </div>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return "No file uploaded"
    file = request.files['file']
    
    try:
        # Preprocess
        processed_data = preprocess_image(file.read())
        
        # Predict
        prediction = model.predict(processed_data)[0][0]
        
        # Logic Label (Threshold 0.5)
        label = "Day (Siang)" if prediction > 0.5 else "Night (Malam)"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return f"""
        <div style="text-align:center;">
            <h2>Hasil: {label}</h2>
            <p>Confidence: {confidence*100:.2f}%</p>
            <a href="/">Coba Lagi</a>
        </div>
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)