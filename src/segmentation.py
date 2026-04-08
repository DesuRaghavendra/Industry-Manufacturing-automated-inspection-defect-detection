'''
import cv2
import numpy as np

def extract_rois(image):
    _, thresh = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h > 100:  # remove small noise
            roi = image[y:y+h, x:x+w]
            rois.append(roi)

    return rois

'''

import cv2
import numpy as np

def extract_largest_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image  # fallback

    # Select largest contour
    largest_cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_cnt)

    # Optional: filter very small ROI
    if w * h < 100:
        return image  # fallback

    roi = image[y:y+h, x:x+w]
    return roi


def pad_to_square(img):
    h, w, _ = img.shape
    size = max(h, w)

    padded = np.zeros((size, size, 3), dtype=img.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return padded