from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model('modelo_entrenado.keras')

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logging.info('Recibiendo solicitud de predicción...')
    
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

        logging.info(f'Salidas del modelo: {prediction}')

        if len(prediction[0]) != 2:
            return jsonify({"error": "El modelo no tiene la salida esperada."}), 500

        healthy_confidence, disease_confidence = prediction[0]
        predicted_class = 'Enfermo' if disease_confidence > healthy_confidence else 'Saludable'
        confidence = max(healthy_confidence, disease_confidence)

        os.remove(img_path)

        logging.info(f'Predicción: {predicted_class}, Confianza: {confidence}')
        
        confidence_message = ""
        if confidence >= 0.95:
            confidence_message = "La predicción es extremadamente confiable."
        elif confidence >= 0.90:
            confidence_message = "Alta confianza en la predicción."
        elif confidence >= 0.80:
            confidence_message = "Confianza moderada en la predicción."
        elif confidence >= 0.70:
            confidence_message = "Confianza baja, revisar la imagen."
        else:
            confidence_message = "Confianza muy baja, la imagen requiere revisión cuidadosa."

        return jsonify({
            "prediction": predicted_class,
            "confidence": str(confidence),
            "confidence_message": confidence_message
        }), 200

    except Exception as e:
        logging.error(f'Error durante la predicción: {str(e)}')
        return jsonify({"error": "Error en el procesamiento de la imagen."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))