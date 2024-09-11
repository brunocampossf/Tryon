import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def filter_images_by_size(images, min_size=768):
    output = []

    for img in tqdm(images, total=len(images)):
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        
        h, w, c = x.shape
        if h > min_size and w > min_size:
            output.append(img)
    
    return output

def filter_images_by_portrait(images):
    output = []

    for img in tqdm(images, total=len(images)):
        x = cv2.imread(img, cv2.IMREAD_COLOR)

        h, w, c = x.shape
        if h > w:
            output.append(img)
    
    return output

def save_images(images, save_dir, size=(768, 512)):
    idx = 1

    for path in tqdm(images, total=len(images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)

        x = cv2.resize(x, (size[1], size[0]))

        print(x.shape)
        cv2.imwrite(f"{save_dir}/{idx}.jpg", x)
        idx += 1

tryon = glob("tryon/images/*.jpg")
print("Initial images: ", len(tryon))
for img in tryon:
    x = cv2.imread(img, cv2.IMREAD_COLOR)
    print(x.shape)


# print("Initial images: ", len(tryon))



# save_images(tryon, "tryon")

# clean_images = glob("clean_images/*.jpg")
# print("Initial images: ", len(clean_images))
# for img in clean_images:
#     x = cv2.imread(img, cv2.IMREAD_COLOR)
#     print(x.shape)

# raw_images = glob("raw_images/*.jpg")
# print("Initial images: ", len(raw_images))

# output = filter_images_by_size(raw_images, min_size=512)
# print("Filter by size: ", len(output))

# output = filter_images_by_portrait(raw_images)
# print("Filter by portrait: ", len(output))

# save_images(output, "clean_images")