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