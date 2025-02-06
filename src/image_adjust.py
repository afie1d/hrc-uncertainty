import numpy as np
import cv2
import random


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
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        cpy = color_jitter(cpy, brightness, contrast, saturation)

        angle = random.uniform(-15, 15)
        cpy = rotate(cpy, angle)

        scale_factor = random.uniform(0.8, 1.2)
        cpy = scale(cpy, scale_factor)

        transformed.append(cpy)

    return transformed

# ------------------------------ Semantic Transforms ----------------------------------------


def blur(img, y1, y2, x1, x2, kernel_size=(63, 63)):
    cpy = img.copy()
    roi = cpy[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, kernel_size, sigmaX=20)
    cpy[y1:y2, x1:x2] = blurred_roi
    return cpy

def random_blur(img, num_imgs=16, patch_size=(256, 256)):
    blurred = []

    for _ in range(num_imgs):
        x1 = np.random.randint(0, img.shape[1])
        y1 = np.random.randint(0, img.shape[0])
        y2 = y1 + patch_size[1]
        x2 = x1 + patch_size[0]

        b = blur(img, y1, y2, x1, x2)
        blurred.append(b)

    return blurred   


def systematic_blur(img, num_rows=4, num_columns=4, patch_size=(256, 256)):
    blurred = []
    (h, w) = img.shape[:2]
    x_spacing = w // num_columns
    y_spacing = h // num_rows

    for r in range(num_rows):
        for c in range(num_columns):
            x1 = min(w - patch_size[1], max(0, x_spacing // 2 + c * patch_size[1] - patch_size[1] // 2))
            y1 = min(h - patch_size[0], max(0, y_spacing // 2 + r * patch_size[0] - patch_size[0] // 2))
            x2 = x1 + patch_size[1]
            y2 = y1 + patch_size[0]
            
            try:
               b = blur(img, y1, y2, x1, x2)
               blurred.append(b)
            except:
                raise BaseException(f"Error occurred. Patch to be blurred: {(y1, y2, x1, x2)}, image shape: {img.shape}")

    return blurred

def load_det_model():
    prototxt_path = "../models/MobileNet/deploy.prototxt"
    model_path = "../models/MobileNet/mobilenet_iter_73000.caffemodel"
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def object_blur(img, detection_model, conf_threshold=0.5):

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    detection_model.setInput(blob)
    detections = detection_model.forward()
    
    blurred = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            b = blur(img, y1, y2, x1, x2)
            blurred.append(b)

    return blurred
