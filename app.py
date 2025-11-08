from flask import Flask, render_template, request, url_for, redirect, session, make_response
import os
from werkzeug.utils import secure_filename
import numpy as np
from preprocess import process_ecg_file
from model_utils import load_models, get_ensemble_prediction
from visualize import plot_ecg_signal
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable file caching

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'dat', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure static directory exists for plots
os.makedirs('static/plots', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_advice(prediction_result, probability):
    """
    Generate actionable advice based on prediction result and probability.
    """
    if prediction_result == "Normal":
        return {
            'urgency': 'Low',
            'explanation': 'Your ECG signal appears to be within normal parameters. However, this analysis is based on AI interpretation and should be verified by a healthcare professional.',
            'actions': [
                'Continue maintaining a healthy lifestyle with regular exercise and balanced diet',
                'Schedule regular annual health check-ups with your primary care physician',
                'Monitor your heart health if you have risk factors (family history, high blood pressure, etc.)',
                'Keep this ECG record for your medical file and discuss with your doctor during your next visit'
            ]
        }
    else:
        # Abnormal prediction
        if probability >= 0.8:
            urgency = 'High'
            explanation = f'Your ECG signal shows strong indicators of potential arrhythmia or heart rhythm abnormalities (confidence: {probability*100:.1f}%). This requires immediate medical attention.'
            actions = [
                '**Seek immediate medical attention** - Contact your healthcare provider or visit an emergency room if experiencing symptoms',
                'Do not ignore symptoms like chest pain, shortness of breath, dizziness, or fainting',
                'Bring this ECG analysis and the original ECG file to your doctor for comprehensive evaluation',
                'Follow up with a cardiologist for detailed cardiac assessment and appropriate treatment',
                'Avoid strenuous activities until cleared by a medical professional',
                'If experiencing severe symptoms, call emergency services (911) immediately'
            ]
        elif probability >= 0.6:
            urgency = 'Moderate'
            explanation = f'Your ECG signal shows signs of potential abnormalities (confidence: {probability*100:.1f}%). While not immediately critical, medical evaluation is recommended.'
            actions = [
                'Schedule an appointment with your primary care physician or cardiologist within 1-2 weeks',
                'Monitor for symptoms such as irregular heartbeat, fatigue, or chest discomfort',
                'Bring this ECG analysis and the original ECG file to your doctor',
                'Maintain a record of any symptoms you experience (when they occur, duration, triggers)',
                'Avoid activities that seem to trigger symptoms until evaluated by a healthcare provider',
                'Consider lifestyle modifications: reduce stress, maintain healthy diet, and follow doctor\'s recommendations'
            ]
        else:
            urgency = 'Moderate'
            explanation = f'Your ECG signal shows some unusual patterns (confidence: {probability*100:.1f}%). While the confidence is moderate, it\'s advisable to have it reviewed by a healthcare professional.'
            actions = [
                'Schedule a consultation with your healthcare provider to review this ECG analysis',
                'Discuss your medical history and any symptoms you may be experiencing',
                'Bring this ECG analysis and the original ECG file for professional interpretation',
                'Consider additional cardiac testing if recommended by your doctor',
                'Maintain regular follow-ups to monitor your heart health',
                'Continue practicing heart-healthy habits: exercise, balanced diet, stress management'
            ]
        
        return {
            'urgency': urgency,
            'explanation': explanation,
            'actions': actions
        }

# Load models at startup
try:
    models = load_models()
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    models = None

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize all variables
    prediction = None
    error_message = None
    ecg_plot_url = None
    probability = None
    advice = None
    
    if request.method == 'POST':
        # Check if models are loaded
        if models is None:
            error_message = "Error: Models not loaded. Please check server configuration."
            logger.error("Models not loaded")
            return render_template('index.html', 
                                 prediction=prediction, 
                                 error=error_message, 
                                 ecg_plot_url=ecg_plot_url, 
                                 probability=probability, 
                                 advice=advice)
        
        # Check if a file was uploaded
        if 'file' not in request.files:
            error_message = "No file selected. Please upload a valid ECG file."
            logger.warning("No file part in request")
            return render_template('index.html', 
                                 prediction=prediction, 
                                 error=error_message, 
                                 ecg_plot_url=ecg_plot_url, 
                                 probability=probability, 
                                 advice=advice)
            
        file = request.files['file']
        if file.filename == '':
            error_message = "No file selected. Please upload a valid ECG file."
            logger.warning("No selected file")
            return render_template('index.html', 
                                 prediction=prediction, 
                                 error=error_message, 
                                 ecg_plot_url=ecg_plot_url, 
                                 probability=probability, 
                                 advice=advice)
            
        if file and allowed_file(file.filename):
            filepath = None
            plot_path = None
            try:
                filename = secure_filename(file.filename)
                if not filename:
                    error_message = "Invalid file name. Please upload a valid ECG file."
                    logger.warning("Invalid filename after securing")
                    return render_template('index.html', 
                                         prediction=prediction, 
                                         error=error_message, 
                                         ecg_plot_url=ecg_plot_url, 
                                         probability=probability, 
                                         advice=advice)
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"File saved successfully: {filepath}")
                
                # Process the ECG signal
                logger.info("Processing ECG signal...")
                processed_signal, raw_signal = process_ecg_file(filepath, return_raw=True, target_length=1000)
                logger.info(f"Signal processed successfully. Shape: {processed_signal.shape}")
                
                # Generate ECG visualization
                try:
                    plot_filename = f"ecg_{uuid.uuid4().hex[:8]}.png"
                    plot_path = os.path.join('static', 'plots', plot_filename)
                    plot_ecg_signal(raw_signal, plot_path, title="ECG Signal Analysis")
                    ecg_plot_url = url_for('static', filename=f'plots/{plot_filename}')
                    logger.info(f"ECG plot generated: {ecg_plot_url}")
                except Exception as e:
                    logger.warning(f"Could not generate ECG plot: {str(e)}")
                    ecg_plot_url = None
                
                # Get prediction
                logger.info("Making prediction...")
                ensemble_prob, dl_prob, xgb_prob, final_prediction = get_ensemble_prediction(processed_signal, models)
                probability = float(ensemble_prob)
                
                # Format prediction result with threshold = 0.5
                if final_prediction == 1:
                    prediction_result = "Abnormal"
                    prediction = "Abnormal (Possible arrhythmia detected)"
                else:
                    prediction_result = "Normal"
                    prediction = "Normal"
                
                logger.info(f"Final prediction: {prediction} (probability: {probability:.6f})")
                
                # Generate actionable advice
                advice = generate_advice(prediction_result, probability)
                logger.info(f"Generated advice with urgency: {advice['urgency']}")
                
                # Store in session for POST-redirect-GET pattern
                session['prediction'] = prediction
                session['probability'] = probability
                session['ecg_plot_url'] = ecg_plot_url
                session['advice'] = advice
                session['dl_prob'] = float(dl_prob)
                session['xgb_prob'] = float(xgb_prob)
                
                # Use POST-redirect-GET pattern to prevent resubmission on refresh
                return redirect(url_for('index'))
                
            except ValueError as e:
                error_message = f"Invalid file format. Please upload a valid ECG file (.dat or .csv). Error: {str(e)}"
                logger.error(f"ValueError processing file: {str(e)}", exc_info=True)
            except RuntimeError as e:
                error_message = f"Error during prediction: {str(e)}"
                logger.error(f"RuntimeError during prediction: {str(e)}", exc_info=True)
            except Exception as e:
                error_message = f"Error processing file: {str(e)}"
                logger.error(f"Unexpected error processing file: {str(e)}", exc_info=True)
            finally:
                # Clean up uploaded file
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logger.info("Temporary file removed")
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file: {str(e)}")
                
                # Clean up plot file if there was an error and plot was created
                if error_message and plot_path and os.path.exists(plot_path):
                    try:
                        os.remove(plot_path)
                        logger.info("Plot file removed due to error")
                    except Exception as e:
                        logger.warning(f"Could not remove plot file: {str(e)}")
        else:
            error_message = "Invalid file format. Please upload a valid ECG file (.dat or .csv)."
            logger.warning(f"Invalid file format: {file.filename if file else 'None'}")
    
    # Handle GET requests and display results from session
    if request.method == 'GET':
        # Check if there are results in session (from POST-redirect-GET)
        if 'prediction' in session:
            prediction = session.pop('prediction', None)
            probability = session.pop('probability', None)
            ecg_plot_url = session.pop('ecg_plot_url', None)
            advice = session.pop('advice', None)
            dl_prob = session.pop('dl_prob', None)
            xgb_prob = session.pop('xgb_prob', None)
            error_message = None
            
            logger.info(f"Displaying results from session: {prediction}")
    
    # Render template for both GET and POST (when not redirecting)
    response = make_response(render_template('index.html', 
                         prediction=prediction, 
                         error=error_message, 
                         ecg_plot_url=ecg_plot_url, 
                         probability=probability, 
                         advice=advice))
    
    # Add cache control headers to prevent browser caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', 
                         prediction=None, 
                         error="File too large. Please upload a file smaller than 16MB.", 
                         ecg_plot_url=None, 
                         probability=None, 
                         advice=None), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)