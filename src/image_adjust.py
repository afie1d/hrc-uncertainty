import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
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


def non_semantic_transform(img, n=16, scale_range=(0.8, 0.12), rotation=15):
    transformed = []

    for _ in range(n):
        cpy = img.copy()
        brightness = random.uniform(scale_range[0], scale_range[1])
        contrast = random.uniform(scale_range[0], scale_range[1])
        saturation = random.uniform(scale_range[0], scale_range[1])
        cpy = color_jitter(cpy, brightness, contrast, saturation)

        angle = random.uniform(-rotation, rotation)
        cpy = rotate(cpy, angle)

        scale_factor = random.uniform(scale_range[0], scale_range[1])
        cpy = scale(cpy, scale_factor)

        transformed.append(cpy)

    return transformed

# ------------------------------ Semantic Transforms ----------------------------------------


def blur(img, y1, y2, x1, x2, kernel_size=(127, 127), invert=False):
    cpy = img.copy()
    roi = cpy[y1:y2, x1:x2]

    if invert:
        cpy = cv2.GaussianBlur(cpy, kernel_size, sigmaX=40)
        cpy[y1:y2, x1:x2] = roi
    else:
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, sigmaX=40)
        cpy[y1:y2, x1:x2] = blurred_roi

    return cpy

def random_blur(img, num_imgs=16, patch_size=(256, 256), invert=False):
    blurred = []

    for _ in range(num_imgs):
        x1 = np.random.randint(0, img.shape[1])
        y1 = np.random.randint(0, img.shape[0])
        y2 = y1 + patch_size[1]
        x2 = x1 + patch_size[0]

        b = blur(img, y1, y2, x1, x2, invert=invert)
        blurred.append(b)

    return blurred   


def systematic_blur(img, num_rows=4, num_columns=4, invert=False):
    blurred = []
    (h, w) = img.shape[:2] 
    x_spacing = w // num_columns
    y_spacing = h // num_rows
    patch_lenth = min(x_spacing, y_spacing)
    patch_size = (patch_lenth, patch_lenth)

    for r in range(num_rows):
        for c in range(num_columns):
            x1 = min(w - patch_size[1], max(0, x_spacing // 2 + c * patch_size[1] - patch_size[1] // 2))
            y1 = min(h - patch_size[0], max(0, y_spacing // 2 + r * patch_size[0] - patch_size[0] // 2))
            x2 = x1 + patch_size[1]
            y2 = y1 + patch_size[0]
            
            try:
               b = blur(img, y1, y2, x1, x2, invert=invert)
               blurred.append(b)
            except:
                raise BaseException(f"Error occurred. Patch to be blurred: {(y1, y2, x1, x2)}, image shape: {img.shape}")

    return blurred

def load_det_model():
    sam_checkpoint = "../models/SAM/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    return SamAutomaticMaskGenerator(sam)

def object_blur(img, detection_model, conf_threshold=0.98, invert=False, max_objects=16):
    blurred = []
    
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # generate masks
    masks = detection_model.generate(img_rgb)
    masks = [m for m in masks if m['stability_score'] > conf_threshold]

    if len(masks) > max_objects:
        sorted_masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        masks = sorted_masks[:max_objects]

    for mask in masks:
        segmentation = mask["segmentation"]
        y_indices, x_indices = np.where(segmentation)
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()

        b = blur(img, y1, y2, x1, x2, invert=invert)
        blurred.append(b)

    return blurred

