import numpy as np
import cv2
import json

f = open("masks.json", "r")
data = json.load(f)
data = data["_via_img_metadata"]

img_dir = "Segmentation/images"
mask_dir = "Segmentation/masks"

for key, value in data.items():
    filename = value["filename"]

    img_path = f"{img_dir}/{filename}"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    mask = np.zeros((h, w))

    regions = value["regions"]
    for region in regions:
        shape_attributes = region["shape_attributes"]
        x_points = shape_attributes["all_points_x"]
        y_points = shape_attributes["all_points_y"]
        
        contours = []
        for x, y in zip(x_points, y_points):
            contours.append((x, y))
        contours = np.array(contours)
        cv2.drawContours(mask, [contours], -1, 255, -1)

    cv2.imwrite(f"{mask_dir}/{filename}", mask)