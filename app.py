import os
import importlib.util
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__)
app.config['DATASET_UPLOAD_FOLDER'] = 'training_dataset'
app.config['MODEL_PATH'] = 'model/model.pt'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# Store file paths for the dataset
dataset_file_path = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset_file():
    global dataset_file_path
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Clear the dataset folder
        folder = app.config['DATASET_UPLOAD_FOLDER']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Save the new file
        filename = secure_filename(file.filename)
        dataset_file_path = os.path.join(folder, filename)
        file.save(dataset_file_path)
        return jsonify({"success": True, "filename": filename}), 200
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/run_forecast', methods=['GET'])
def run_forecast():
    global dataset_file_path

    if not dataset_file_path:
        return jsonify({"error": "Dataset must be uploaded"}), 400
    
    try:
        # Run the model for prediction
        predictions = run_lstm_and_get_predictions(dataset_file_path)
        return jsonify({"success": True, "predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from model import LSTM_train_eva
def run_lstm_and_get_predictions(dataset_file_path):
    # spec = importlib.util.spec_from_file_location("LSTM_train_eva", "model/LSTM_train_eva.py")
    # lstm_train_eva = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(lstm_train_eva)
    
    
    # Ensure the main function of LSTM_train_eva.py handles the training and prediction
    predictions = LSTM_train_eva.main(dataset_file_path)
    
    # Convert predictions to a list of dictionaries
    result = []
    for i, prediction in enumerate(predictions):
        result.append({"original_data": i, "prediction": prediction})  # Adjust the keys as per your requirement
    
    return result

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error: " + str(error)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found: " + str(error)}), 404

if __name__ == '__main__':
    app.run(debug=True)