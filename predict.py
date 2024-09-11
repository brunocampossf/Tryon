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

height = 768
width = 512

# dataset_path = "dataset/non-aug/test"
# save_path = "prediction/non-aug"
# model_path = "files/non-aug/unet-non-aug.h5"

dataset_path = "tryon"
save_path = "tryon/mask"
model_path = "files/aug/unet-aug.h5"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir(save_path)

model = tf.keras.models.load_model(model_path)

test_x = sorted(glob(os.path.join(dataset_path, "images", "*")))

time_taken = []

for x in tqdm(test_x):
    name = x.split("/")[-1]

    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = x/255.0
    x = np.expand_dims(x, axis=0)

    start_time = time.time()
    p = model.predict(x)[0]
    total_time = time.time() - start_time
    time_taken.append(total_time)

    p = p > 0.5
    p = p * 255

    cv2.imwrite(os.path.join(save_path, name), p)

# mean_time = np.mean(time_taken)
# mean_fps = 1 / mean_time