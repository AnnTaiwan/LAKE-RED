import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random
import cv2
import os
from ultralytics import YOLO
'''
Must run this code at this dir: D:/NSYSU_Fourth_grade/cv_final_project/LAKE-RED/create_mask/src
'''
SRC_DIR = "../../../data/obstacle_dataset/images2" # original images
DEST_DIR = "../../../data/obstacle_dataset/masks2/" # mask images

def do_segmentation(path, model, conf, dst_dir=DEST_DIR):
    img_bgr = cv2.imread(path) # 0-255

    # a white image, waited  for fill some black pixel that is object
    gray_white_image = np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * 255

    # Predict with the model
    # initially, resize image into 640*640
    results = model.predict(img_bgr, conf=conf)  # predict on an image, it can expect BGR opencv image
    try: 
        # Access the results
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask]) 
                cv2.fillPoly(gray_white_image, points, 0) # fill black
        
        cv2.imwrite(dst_dir + path.split('\\')[-1].split('.')[0] + ".png", gray_white_image)
    except Exception as e:
        print(path)
        print(f"\033[91mError: {e}\033[0m")

if __name__ == "__main__":
    if not os.path.exists(SRC_DIR) or not os.path.exists(DEST_DIR):
        print("ERROR: Images folder doesn't exist.")
        exit(1)
    # Load a model
    model = YOLO("../yolo_segmentation_ckpt/yolo11l-seg.pt")
    conf = 0.5

    for file_name in os.listdir(SRC_DIR):
        print(f"\033[93mProcess {file_name}...\033[0m")
        do_segmentation(os.path.join(SRC_DIR, file_name), model, conf, DEST_DIR)

    print(f"All masks are created in {DEST_DIR}")