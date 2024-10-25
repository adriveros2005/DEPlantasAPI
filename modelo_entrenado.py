"""modelo_entrenado.py"""

import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

base_dir = "PlantVillage-Dataset/raw/color/"

healthy_folders = ['Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___healthy', 'Grape___healthy', 'Peach___healthy', 'Pepper,_bell___healthy',
                   'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Strawberry___healthy',
                   'Tomato___healthy']
disease_folders = [folder for folder in os.listdir(base_dir) if "healthy" not in folder and folder != '.git']

sampled_data_dir = 'sample_data'
os.makedirs(sampled_data_dir, exist_ok=True)

image_source = {}

def sample_images(folder_list, label, sample_size=1000):
    img_paths = []
    labels = []

    for folder in folder_list:
        folder_path = os.path.join(base_dir, folder)
        images = os.listdir(folder_path)
        sampled_images = random.sample(images, min(sample_size, len(images)))

        for img in sampled_images:
            img_src = os.path.join(folder_path, img)
            img_dst = os.path.join(sampled_data_dir, img)
            shutil.copy(img_src, img_dst)
            img_paths.append(img_dst)
            labels.append(label)
            image_source[img_dst] = folder 

    return img_paths, labels

healthy_img_paths, healthy_labels = sample_images(healthy_folders, label=0, sample_size=1000)
disease_img_paths, disease_labels = sample_images(disease_folders, label=1, sample_size=1000)

all_img_paths = healthy_img_paths + disease_img_paths
all_labels = healthy_labels + disease_labels

train_paths, test_paths, train_labels, test_labels = train_test_split(all_img_paths, all_labels, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.25, random_state=42)

datagen = ImageDataGenerator(rescale=1./255)

def create_generator(image_paths, labels, batch_size=32):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            images = []

            for img_path in batch_paths:
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img)
                except Exception as e:
                    print(f"Error cargando la imagen {img_path}: {e}")

            images = np.array(images) / 255.0
            batch_labels = to_categorical(batch_labels, num_classes=2)

            yield images, batch_labels

train_gen = create_generator(train_paths, train_labels)
val_gen = create_generator(val_paths, val_labels)
test_gen = create_generator(test_paths, test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 20
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_gen, steps=len(test_paths) // batch_size)
print(f"Precisión en el set de prueba: {test_acc * 100:.2f}%")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Precisión en Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión en Validación')
plt.title('Curva de Precisión por Épocas')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

model.save('modelo_entrenado.keras')