from flask import Flask, render_template, request, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
from preprocess import process_ecg_file
from model_utils import load_models, get_ensemble_prediction
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'dat', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load models at startup
try:
    models = load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    models = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            logger.warning("No file part in request")
            return render_template('index.html')
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            logger.warning("No selected file")
            return render_template('index.html')
            
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File saved successfully: {filepath}")
                
                # Process the ECG signal
                logger.info("Processing ECG signal...")
                processed_signal = process_ecg_file(filepath)
                logger.info("Signal processed successfully")
                
                # Get prediction
                logger.info("Making prediction...")
                pred_result = get_ensemble_prediction(processed_signal, models)
                logger.info(f"Prediction result: {pred_result}")
                
                # Format prediction result
                if pred_result > 0.5:
                    prediction = "Abnormal (Possible arrhythmia detected)"
                else:
                    prediction = "Normal"
                logger.info(f"Final prediction: {prediction}")
                
                # Clean up uploaded file
                os.remove(filepath)
                logger.info("Temporary file removed")
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}", exc_info=True)
                flash(f"Error processing file: {str(e)}")
                return render_template('index.html')
        else:
            flash('Invalid file format. Please upload a valid ECG file (.dat or .csv)')
            logger.warning(f"Invalid file format: {file.filename}")
            return render_template('index.html')
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)