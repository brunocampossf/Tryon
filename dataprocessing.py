import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, CoarseDropout, RandomBrightnessContrast

def load_dataset(path):
    images = sorted(glob(os.path.join(path, "images", "*")))
    masks = sorted(glob(os.path.join(path, "masks", "*")))
    return images, masks

def split_dataset(images, masks, split=0.2):
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
    
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)


    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_dataset(images, masks, save_dir, augement=False):
    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")[0]
        
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augement == True:
            aug = HorizontalFlip(p=1)
            augemented = aug(image=x, mask=y)
            x1 = augemented["image"]
            y1 = augemented["mask"]

            aug = CoarseDropout(p=1, max_holes=10, max_height=32, max_width=32)
            augemented = aug(image=x, mask=y)
            x2 = augemented["image"]
            y2 = augemented["mask"]
            
            aug = RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1)
            augemented = aug(image=x, mask=y)
            x3 = augemented["image"]
            y3 = augemented["mask"]
            
            aug = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1)
            augemented = aug(image=x, mask=y)
            x4 = augemented["image"]
            y4 = augemented["mask"]

            aug_x = [x, x1, x2, x3, x4]
            aug_y = [y, y1, y2, y3, y4]
        else:
            aug_x = [x]
            aug_y = [y]

        idx = 0
        for ax, ay in zip(aug_x, aug_y):
            if augement:
                aug_name = f"{name}_{idx}.jpg"
                
            else:
                aug_name = f"{name}.jpg"

            save_image_path = os.path.join(save_dir, "images", aug_name)
            save_mask_path = os.path.join(save_dir, "masks", aug_name)

            cv2.imwrite(save_image_path, ax)
            cv2.imwrite(save_mask_path, ay)

            idx += 1


    
dataset_path = "Segmentation"
images, masks = load_dataset(dataset_path)

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = split_dataset(images, masks, split=0.2)

save_dir_nonaug = os.path.join("dataset", "non-aug")
for item in ["train", "valid", "test"]:
    create_dir(os.path.join(save_dir_nonaug, item, "images"))
    create_dir(os.path.join(save_dir_nonaug, item, "masks"))

save_dataset(train_x, train_y, os.path.join(save_dir_nonaug, "train"), augement=False)
save_dataset(valid_x, valid_y, os.path.join(save_dir_nonaug, "valid"), augement=False)
save_dataset(test_x, test_y, os.path.join(save_dir_nonaug, "test"), augement=False)

save_dir_aug = os.path.join("dataset", "aug")
for item in ["train", "valid", "test"]:
    create_dir(os.path.join(save_dir_aug, item, "images"))
    create_dir(os.path.join(save_dir_aug, item, "masks"))

save_dataset(train_x, train_y, os.path.join(save_dir_aug, "train"), augement=True)
save_dataset(valid_x, valid_y, os.path.join(save_dir_aug, "valid"), augement=False)
save_dataset(test_x, test_y, os.path.join(save_dir_aug, "test"), augement=False)