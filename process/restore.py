import cv2
import numpy as np

def restore_image(np_image):
    # 1. Konversi ke grayscale
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # 3. Equalisasi histogram
    equalized = cv2.equalizeHist(denoised)

    # 4. Deteksi goresan terang dan gelap
    _, light_scratches = cv2.threshold(equalized, 245, 255, cv2.THRESH_BINARY)
    _, dark_spots = cv2.threshold(equalized, 10, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_or(light_scratches, dark_spots)

    # 5. Inpainting
    inpainted = cv2.inpaint(equalized, mask, 3, cv2.INPAINT_TELEA)

    # 6. Smoothing
    smooth = cv2.bilateralFilter(inpainted, d=1, sigmaColor=75, sigmaSpace=75)

    # 7. Sharpening (Unsharp Masking)
    blurred = cv2.GaussianBlur(smooth, (9, 9), 10)
    sharpened = cv2.addWeighted(smooth, 1.5, blurred, -0.5, 0)

    # 8. Convert ke BGR
    result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # 9. Encode ke JPG
    _, encimg = cv2.imencode('.jpg', result)
    return encimg

