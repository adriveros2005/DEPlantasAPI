from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Inicializar la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.keras')

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para la predicción con imagen subida por el usuario
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Imagen no válida o no recibida"}), 400
    
    file = request.files['image']
    
    if not file:
        return jsonify({"error": "Imagen no válida o no recibida"}), 400

    try:
        # Guardar la imagen temporalmente
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        
        # Procesar la imagen con el tamaño adecuado
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Realizar la predicción
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = 'Enfermo' if prediction[0][1] > 0.5 else 'Saludable'
        
        # Eliminar la imagen temporal después de la predicción
        os.remove(img_path)

        # Retornar la predicción como respuesta JSON
        return jsonify({"prediction": predicted_class, "confidence": str(confidence)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Correr la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
