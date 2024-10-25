from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

tf.config.set_visible_devices([], 'GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model('modelo_entrenado.keras')

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400
    
    file = request.files['image']
    
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Imagen no válida o formato no soportado"}), 400

    try:
    
        filename = secure_filename(file.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(img_path)
        
        img = load_img(img_path, target_size=(128, 128))  
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = 'Enfermo' if prediction[0][1] > 0.5 else 'Saludable'
        
        os.remove(img_path)

        return jsonify({"prediction": predicted_class, "confidence": str(confidence)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))