import cv2
import numpy as np

def restore_image(np_image):
    # 1. Denoising warna (tanpa menghilangkan tepi tajam)
    denoised = cv2.bilateralFilter(np_image, d=15, sigmaColor=90, sigmaSpace=90)

    # 2. Ubah ke HSV untuk peningkatan kontras pada channel V
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[..., 2] = clahe.apply(v)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Deteksi goresan putih dengan threshold tinggi
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, scratch_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # 4. Inpainting untuk menghapus goresan putih
    inpainted = cv2.inpaint(enhanced, scratch_mask, 3, cv2.INPAINT_TELEA)

    # 5. Optional: sharpening halus
    blurred = cv2.GaussianBlur(inpainted, (0, 0), 3)
    sharpened = cv2.addWeighted(inpainted, 1.5, blurred, -0.5, 0)

    # 6. Encode ke JPEG
    _, encimg = cv2.imencode('.jpg', sharpened)
    return encimg
