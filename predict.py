import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from modules.utils import encode_binary_mask

test_dir = './data/test'

test_data = []

test_filenames = sorted(os.listdir(test_dir))

for filename in test_filenames:
    test = cv2.imread(os.path.join(test_dir, filename))
    test_data.append(test)

X_test = np.array(test_data)

model = load_model('./models/hubmap_model_{}.h5')

predictions = model.predict(X_test)

predictions = (predictions > 0.5).astype(np.uint8) * 255

folder = './data/predictions'
if not os.path.exists(folder):
    os.mkdir(folder)

submission_table = pd.DataFrame(columns = ['id', 'height', 'width', 'prediction_string'])
for i, img_name in enumerate(test_filenames):
    img = predictions[i]
    img_bool = img == 1
    base64_encode = encode_binary_mask(img_bool)
    prediction_string = '0 ' + '1.0 ' + base64_encode
    new_row = {'id': img_name, 'height': img.shape[0], 'width': img.shape[1], 'predictions_string': prediction_string}
    submission_table.append(new_row)
    cv2.imwrite(f'./data/predictions/{img_name}.tif', img)

submission_table.to_csv('submission.csv')