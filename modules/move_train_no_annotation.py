import pandas as pd
import shutil
import os

# Leer el archivo CSV
df = pd.read_csv('./data/tile_meta.csv')

# Crear la carpeta si no existe
if not os.path.exists('./data/train_no_annotation'):
    os.makedirs('./data/train_no_annotation')

# Filtrar las imágenes que pertenecen al grupo 3
group_3_images = df[df['dataset'] == 3]['id']

# Mover las imágenes al directorio 'train_no_annotation'
for image in group_3_images:
    print(f'./data/train/{image}.tif')
    # Asegúrate de que la imagen exista antes de moverla
    if os.path.isfile(f'./data/train/{image}.tif'):
        shutil.move(f'./data/train/{image}.tif', f'./data/train_no_annotation/{image}.tif')
