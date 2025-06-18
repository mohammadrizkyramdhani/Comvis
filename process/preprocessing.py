import cv2
import numpy as np

def preprocessing_image(np_image):
    # 1. Konversi BGR ke RGB
    rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    # 2. Ekstraksi channel hijau
    green_channel = rgb[:, :, 1]

    # 3. Konversi ke grayscale
    gray = green_channel 

    # 4. CLAHE (peningkatan kontras lokal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 5. Bilateral Filter (denoising + edge-preserving)
    smooth = cv2.bilateralFilter(enhanced, d=3, sigmaColor=25, sigmaSpace=25)

    # 6. Konversi ke format 3-channel BGR (untuk kompatibilitas)
    result = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)

    # 7. Encode ke JPEG
    _, encimg = cv2.imencode('.jpg', result)
    return encimg
