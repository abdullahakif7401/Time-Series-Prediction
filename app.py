import os
import importlib.util
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__)
app.config['TRAINING_UPLOAD_FOLDER'] = 'training_dataset'
app.config['PREDICTION_UPLOAD_FOLDER'] = 'prediction_dataset'
app.config['MODEL_PATH'] = 'model/model.pt'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# Store file paths for training and prediction datasets
training_file_path = None
prediction_file_path = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_training', methods=['POST'])
def upload_training_file():
    return handle_file_upload('training')

@app.route('/upload_prediction', methods=['POST'])
def upload_prediction_file():
    return handle_file_upload('prediction')

def handle_file_upload(upload_type):
    global training_file_path, prediction_file_path

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if upload_type == 'training':
            folder = app.config['TRAINING_UPLOAD_FOLDER']
            training_file_path = os.path.join(folder, filename)
        else:
            folder = app.config['PREDICTION_UPLOAD_FOLDER']
            prediction_file_path = os.path.join(folder, filename)
        
        # Clear the folder before saving the new file
        clear_folder(folder)

        # Save the new file
        file.save(os.path.join(folder, filename))
        return jsonify({"success": True, "filename": filename}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/run_forecast', methods=['GET'])
def run_forecast():
    global training_file_path, prediction_file_path

    if not training_file_path or not prediction_file_path:
        return jsonify({"error": "Both training and prediction datasets must be uploaded"}), 400
    
    try:
        # Train the model
        train = train_model(training_file_path)
        # Run the model for prediction
        predictions = predict_model(prediction_file_path)
        return jsonify({"success": True, "training": train, "predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def train_model(training_file_path):
    spec = importlib.util.spec_from_file_location("LSTM_train_eva", "model/LSTM_train_eva.py")
    lstm_train_eva = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_train_eva)
    lstm_train_eva.main(training_file_path)

def predict_model(prediction_file_path):
    spec = importlib.util.spec_from_file_location("LSTM_train_eva", "model/LSTM_train_eva.py")
    lstm_train_eva = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_train_eva)
    predictions = lstm_train_eva.predict(prediction_file_path)
    return predictions

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error: " + str(error)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found: " + str(error)}), 404

if __name__ == '__main__':
    # Ensure upload folders exist
    os.makedirs(app.config['TRAINING_UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PREDICTION_UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
