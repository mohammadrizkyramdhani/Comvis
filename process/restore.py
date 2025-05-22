import cv2
import numpy as np

def restore_image(np_image):
    # 1. Convert ke grayscale (karena gambar jadul umumnya grayscale)
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    # 2. Denoising untuk kurangi noise awal
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # 3. Histogram Equalization untuk kontras
    equalized = cv2.equalizeHist(denoised)

    # 4. Sharpening dengan unsharp masking
    gaussian = cv2.GaussianBlur(equalized, (9, 9), 10.0)
    sharpened = cv2.addWeighted(equalized, 1.5, gaussian, -0.5, 0)

    # 5. Inpainting untuk goresan (gunakan deteksi tepi dan threshold)
    _, thresh = cv2.threshold(sharpened, 245, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(sharpened, thresh, 3, cv2.INPAINT_TELEA)

    # 6. Convert kembali ke 3-channel untuk ditampilkan
    restored_rgb = cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR)

    # Encode untuk dikirim ke frontend
    _, encimg = cv2.imencode('.jpg', restored_rgb)
    return encimg
