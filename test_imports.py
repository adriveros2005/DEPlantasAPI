# test_imports.py
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    print("Importaciones exitosas.")
except ImportError as e:
    print(f"Error de importación: {e}")
