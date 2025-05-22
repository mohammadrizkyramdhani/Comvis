import cv2
import numpy as np

def compress_image(np_image):
    height, width = np_image.shape[:2]
    resized = cv2.resize(np_image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, encimg = cv2.imencode('.jpg', resized, encode_param)
    return encimg
