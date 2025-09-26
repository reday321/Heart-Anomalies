from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)
MODEL_PATH = 'ecg_model.h5'
model = load_model(MODEL_PATH)

# Define class labels (dataset এর সাথে মিলাতে হবে)
class_labels = ['Normal', 'AFib', 'PVC']

def preprocess_img(file_path):
    # Read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'})
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    img = preprocess_img(file_path)
    preds = model.predict(img)[0]
    
    results = []
    for i, label in enumerate(class_labels):
        results.append({
            'title': label,
            'confidence': f"{round(preds[i]*100,2)}%",
            'detail': f"Probability: {round(preds[i]*100,2)}%"
        })

    # Sort descending by confidence
    results = sorted(results, key=lambda x: float(x['confidence'][:-1]), reverse=True)
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
