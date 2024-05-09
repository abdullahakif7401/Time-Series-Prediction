from flask import Flask, send_from_directory, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/upload_page')
def upload_page():
    return send_from_directory('static', 'upload_forecast.html')  # Make sure this file is in the static directory

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify, send_from_directory
# import os
# from werkzeug.utils import secure_filename
# from lstm import process_file  # Ensure lstm.py contains the appropriate functions

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
# app.config['MAX_CONTENT_LENGTH'] = 800 * 1024 * 1024  # Max upload - 800MB
# app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/')
# def index():
#     return send_from_directory('static', 'index.html')

# @app.route('/upload_page')
# def upload_page():
#     return send_from_directory('static', 'upload_forecast.html')  # Ensure this file is in the static directory

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         try:
#             results = process_file(filepath)  # Use the new function
#             return jsonify({'results': results})
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500
#     return jsonify({'error': 'File type not allowed'}), 400

# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])

#     app.run(debug=True)
