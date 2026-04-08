
import cv2
import numpy as np
'''
def dip_preprocess(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Median filtering
    image = cv2.medianBlur(image, 3)

    # Apply CLAHE on each channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image

'''

def dip_preprocess(image):
    # Keep BGR

    # CLAHE (lighting normalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen (enhance defects)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    return image