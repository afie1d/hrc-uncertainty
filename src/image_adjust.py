import tensorflow as tf
import numpy as np
import cv2
import random
import time


# ----------------------------- Non-semantic transforms ------------------------------------

def color_jitter(img, brightness=1.0, contrast=1.0, saturation=1.0):

    # brightness
    img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)

    # saturation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_img[..., 1] *= saturation
    hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)
    img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # contrast
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
    
    return img

def rotate(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_img


def scale(img, scale_factor):
    new_width = int(img.shape[1] * scale_factor)
    new_height = int(img.shape[0] * scale_factor)
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return scaled_img


def non_semantic_transform(img_path, n=16):
    img = cv2.imread(img_path)
    transformed = []

    for _ in range(n):
        cpy = img.copy()
        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        saturation = random.uniform(0.5, 1.5)
        cpy = color_jitter(cpy, brightness, contrast, saturation)

        angle = random.uniform(-15, 15)
        cpy = rotate(cpy, angle)

        scale_factor = random.uniform(0.8, 1.2)
        cpy = scale(cpy, scale_factor)

        transformed.append(cpy)

    return transformed

# ------------------------------ Semantic Transforms ----------------------------------------

def load_det_model():
    model = tf.saved_model.load("../models/efficientdet_d7")
    return model

def blind_blur(img, num_imgs=16, kernel_size=(128, 128)):
    blurred = []

    for _ in range(num_imgs):
        cpy = img.copy()
        x_start = np.random.randint(0, img.shape[1])
        y_start = np.random.randint(0, img.shape[0])

        roi = cpy[y_start:y_start+kernel_size[0], x_start:x_start+kernel_size[1]]
        blurred_roi = cv2.GaussianBlur(roi, (31, 31), sigmaX=20)
        cpy[y_start:y_start+kernel_size[0], x_start:x_start+kernel_size[1]] = blurred_roi

        blurred.append(cpy)

    return blurred   
 
def detect_objects(img, detection_model, conf_threshold=0.5):

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detection_model(input_tensor)
    detection_scores = detections["detection_scores"].numpy()[0]
    detection_boxes = detections["detection_boxes"].numpy()[0]

    masks = []
    for i, score in enumerate(detection_scores):
        if score >= conf_threshold:
            box = detection_boxes[i]
            y1, x1, y2, x2 = int(box[0] * img.shape[0]), int(box[1] * img.shape[1]), int(
                box[2] * img.shape[0]
            ), int(box[3] * img.shape[1])

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            masks.append(mask)

    return masks


def semantic_transform(img_path, n=16):
    img = cv2.imread(img_path)
    return blind_blur(img, num_imgs=n)

# Testing
tfs = semantic_transform('../data/dummy/drift_cars.jpg')

for i, m in enumerate(tfs):
    cv2.imwrite(f'../data/dummy/smeg{i}.jpg', m)