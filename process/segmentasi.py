import cv2
import numpy as np

def region_growing(image, seed_point, threshold=10):
    h, w = image.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    seed_x, seed_y = seed_point
    region_mean = int(image[seed_y, seed_x])
    queue = [(seed_x, seed_y)]

    while queue:
        x, y = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = 1

        pixel_val = int(image[y, x])
        if abs(pixel_val - region_mean) < threshold:
            mask[y, x] = 255
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    queue.append((nx, ny))
    return mask

def apply_watershed_with_gradient(original_img, region_mask):
    kernel = np.ones((5, 5), np.uint8)
    region_mask_closed = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)

    fg = cv2.erode(region_mask_closed, kernel, iterations=2)
    bg = cv2.dilate(region_mask_closed, kernel, iterations=3)
    unknown = cv2.subtract(bg, fg)

    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    gray = original_img if len(original_img.shape) == 2 else cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))

    watershed_input = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_input, markers)

    result = original_img.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    result[markers == -1] = [0, 0, 255]
    return result

def segmentasi_image(np_image):
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    seed = (100, 260)  # disesuaikan dengan lokasi hati
    mask_rg = region_growing(gray, seed, threshold=15)
    result = apply_watershed_with_gradient(gray, mask_rg)
    cv2.circle(result, seed, 5, (0, 255, 0), -1)  # titik seed (hijau)

    _, encoded = cv2.imencode('.jpg', result)
    return encoded.tobytes()
