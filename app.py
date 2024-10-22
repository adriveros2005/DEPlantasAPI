from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import shutil
from git import Repo

# Inicializar la app Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.keras')

# Ruta donde se guardarán las imágenes del dataset
dataset_dir = 'dataset'
os.makedirs(dataset_dir, exist_ok=True)

# URL del repositorio del dataset
repo_url = 'https://github.com/spMohanty/PlantVillage-Dataset.git'
image_folder = os.path.join(dataset_dir, 'PlantVillage-Dataset', 'raw', 'color')

# Función para clonar el dataset
def clone_dataset():
    if not os.path.exists(os.path.join(dataset_dir, 'PlantVillage-Dataset')):
        print("Clonando el dataset desde GitHub...")
        Repo.clone_from(repo_url, dataset_dir)
        print("Dataset clonado.")
    else:
        print("El dataset ya está clonado.")

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
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)
        
        # Procesar la imagen con el tamaño correcto
        img = load_img(img_path, target_size=(128, 128))  # Cambiado a 128x128
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predicción
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = 'Enfermo' if prediction[0][1] > 0.5 else 'Saludable'  # Cambiado a 1 para enfermedad

        os.remove(img_path)  # Eliminar la imagen temporal

        return jsonify({"prediction": predicted_class, "confidence": str(confidence)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para predecir usando imágenes del dataset de GitHub
@app.route('/predict_github', methods=['GET'])
def predict_github():
    try:
        # Clonar el dataset si no está clonado
        clone_dataset()

        # Seleccionar una imagen del dataset clonado
        image_path = os.path.join(image_folder, 'Apple___healthy', '0a3cb91c-25ae-45c4-b85d-899b82646a01___RS_HL 7403.JPG')  # Cambia el nombre del archivo según el dataset

        # Procesar la imagen con el tamaño correcto
        img = load_img(image_path, target_size=(128, 128))  # Cambiado a 128x128
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predicción
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = 'Enfermo' if prediction[0][1] > 0.5 else 'Saludable'  # Cambiado a 1 para enfermedad

        return jsonify({"prediction": predicted_class, "confidence": str(confidence)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Correr la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# Endpoint para predecir usando una imagen del dataset clonado de GitHub
@app.route('/predict_github', methods=['GET'])
def predict_github():
    try:
        # Clonar el dataset si no está clonado
        clone_dataset()

        # Seleccionar una imagen específica del dataset
        image_path = os.path.join(image_folder, 'Apple___healthy', '0a3cb91c-25ae-45c4-b85d-899b82646a01___RS_HL 7403.JPG')  # Cambia el nombre del archivo según tu dataset

        # Procesar la imagen
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Realizar la predicción
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = 'Oídio' if prediction[0][0] > 0.5 else 'Sin enfermedad'

        return jsonify({"prediction": predicted_class, "confidence": str(confidence)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Correr la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)