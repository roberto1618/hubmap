from modules.build_model import BuildModel
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Directorios donde se encuentran tus imágenes y etiquetas
train_dir = './data/train'
label_dir = './data/labels'

# Lista para almacenar tus datos
train_data = []
label_data = []

# Asegurarse de que las imágenes y las etiquetas se emparejan correctamente
image_filenames = sorted(os.listdir(train_dir))
label_filenames = sorted(os.listdir(label_dir))


assert image_filenames == label_filenames, "Las imágenes y las etiquetas no se emparejan correctamente"

# Leer y procesar las imágenes y etiquetas
for filename in image_filenames:
    # Leer la imagen y la etiqueta
    img = cv2.imread(os.path.join(train_dir, filename))
    label = cv2.imread(os.path.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)  # Asegúrate de leer las etiquetas en escala de grises

    # Agregar los datos a las listas
    train_data.append(img)
    label_data.append(label)

# Convertir las listas a arrays de NumPy para que puedan ser usados por tu modelo
X_train = np.array(train_data)
y_train = np.array(label_data)

# Ahora puedes usar train_data y label_data para entrenar tu modelo
model = BuildModel(img_shape = (512, 512, 3), num_classes = 1)
model = model.build_model()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 4, epochs = 10, validation_split = 0.2)

now = datetime.now()
# Formatea como una cadena
timestamp_str = now.strftime("%Y_%m_%d_%H_%M_%S")
# Guarda el modelo con el nombre del archivo que incluye la marca de tiempo
model.save('./models/hubmap_model_{}.h5'.format(timestamp_str))
