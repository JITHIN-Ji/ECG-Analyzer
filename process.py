import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# === 1. Read and Threshold Image ===
image_path = 'abcd.png'
image_color = cv2.imread(image_path)
img_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# === 2. Otsu Thresholding ===
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.title('Otsu Thresholded Image')
plt.show()

# === 3. Histogram and Optimal Thresholding ===
def Hist(img):
    y = np.zeros(256)
    for val in img.flatten():
        y[val] += 1
    x = np.arange(0, 256)
    plt.bar(x, y, color='b', width=5, alpha=0.25)
    plt.title('Histogram')
    plt.show()
    return y

threshold_values = {}
def countPixel(h): return np.sum(h > 0)
def weight(h, s, e): return np.sum(h[s:e])
def mean(h, s, e): return np.sum([i * h[i] for i in range(s, e)]) / weight(h, s, e) if weight(h, s, e) else 0
def variance(h, s, e):
    m = mean(h, s, e)
    w = weight(h, s, e)
    return np.sum([(i - m) ** 2 * h[i] for i in range(s, e)]) / w if w else 0

def compute_threshold(h):
    cnt = countPixel(h)
    for i in range(1, len(h)):
        vb = variance(h, 0, i)
        wb = weight(h, 0, i) / cnt
        vf = variance(h, i, len(h))
        wf = weight(h, i, len(h)) / cnt
        V2w = wb * vb**2 + wf * vf**2
        if not math.isnan(V2w): threshold_values[i] = V2w

def regenerate_img(img, threshold):
    return np.where(img >= threshold, 255, 0).astype(np.uint8)

h = Hist(img_gray)
compute_threshold(h)
opt_thresh = min(threshold_values, key=threshold_values.get)
print("[✅] Optimal threshold:", opt_thresh)

res = regenerate_img(img_gray, opt_thresh)
plt.imshow(res, cmap='gray')
plt.title('Custom Threshold Image')
plt.show()

# === 4. Denoising ===
noisy = np.float32(res) + np.random.randn(*res.shape) * 10
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
dst = cv2.fastNlMeansDenoising(noisy, None, 3, 7, 21)

# === ✅ Use Correct Coordinates from XML (xmin, ymin, xmax, ymax) ===
x1 = 101
y1 = 606
x2 = 1358
y2 = 795
crop_img = dst[y1:y2, x1:x2]


cv2.imwrite('cropped_ecg_leadII.png', crop_img)
print("[✅] Saved cropped image as 'cropped_ecg_leadII.png'")

# === 5. Remove Small Noise Dots ===
for i in range(5, crop_img.shape[0] - 5):
    for j in range(5, crop_img.shape[1] - 5):
        if crop_img[i, j] == 0:
            window = crop_img[i - 5:i + 5, j - 5:j + 5]
            if np.sum(window == 0) < 5:
                crop_img[i, j] = 255

plt.imshow(crop_img, cmap='gray')
plt.title("Cleaned ECG Strip")
plt.show()

# === 6. Trace Digitization ===
signal = []
for col in range(crop_img.shape[1]):
    col_vals = crop_img[:, col]
    black_pixels = np.where(col_vals == 0)[0]
    signal.append(black_pixels[0] if len(black_pixels) > 0 else (signal[-1] if signal else 0))

plt.plot(signal)
plt.title("Extracted Signal (from top)")
plt.grid(True)
plt.show()

# === 7. Export to CSV ===
pd.DataFrame({'Lead_I': signal}).to_csv('digitized_trace.csv', index=False)
print("[✅] Saved digitized_trace.csv")
