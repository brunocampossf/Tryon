import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

model_path = "files/aug/unet-aug.h5"
mask_path = "tryon/mask"
img_path ="tryon/images"
result_path = "tryon/result"

model = tf.keras.models.load_model(model_path)

def load_data(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {image_path}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Erro ao carregar a máscara: {mask_path}")

    image_normalized = image / 255.0

    image_batch = np.expand_dims(image_normalized, axis=0)

    prediction = model.predict(image_batch)
    predicted_mask = np.squeeze(prediction)

    return image, predicted_mask


def tryon(person_img, shirt_img, person_mask, shirt_mask):
    combined = np.copy(person_img)
    
    mask_intersection = (person_mask > 0.5) & (shirt_mask > 0.5)
    
    combined[mask_intersection] = shirt_img[mask_intersection]
    
    return combined

try:
    person_img, person_mask = load_data(os.path.join(img_path, "6.jpg"), os.path.join(mask_path, "6.jpg"))
    shirt_img, shirt_mask = load_data(os.path.join(img_path, "1.jpg"), os.path.join(mask_path, "1.jpg"))
    
    result = tryon(person_img, shirt_img, person_mask, shirt_mask)
    line = np.ones((person_img.shape[0], 10, 3)) * 255
    
    cat_img = np.concatenate([person_img, line, shirt_img, line, result], axis=1)

    result_file = os.path.join(result_path, "tryon_result.jpg")
    cv2.imwrite(result_file, cat_img)

    print(f"Resultado salvo em {result_file}")

except FileNotFoundError as e:
    print(e)

