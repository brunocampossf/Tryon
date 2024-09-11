import os
import numpy as np
import cv2
from tqdm import tqdm

# image_names = sorted(os.listdir("Segmentation/images"))
image_names = sorted(os.listdir("tryon/images"))

for name in tqdm(image_names):
    # image_path = f"Segmentation/images/{name}"
    # mask_path = f"Segmentation/masks/{name}"
    
    image_path = f"tryon/images/{name}"
    mask_path = f"tryon/masks/{name}"

    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    line = np.ones((x.shape[0], 10, 3)) * 255

    f_img = x * (y/255.0)
    b_img = x * (1 - y/255.0)
    
    cat_img = np.concatenate([x, line, y, line, f_img, line, b_img], axis=1)
    cv2.imwrite(f"tryon/cat_img/{name}", cat_img)
