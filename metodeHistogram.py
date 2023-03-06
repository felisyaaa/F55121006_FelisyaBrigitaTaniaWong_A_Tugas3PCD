import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar
img = cv2.imread('bloodCells.jpg')

# konversi gambar ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# membuat histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# mendapatkan nilai pixel terendah dan tertinggi
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

# membuat tipe citra gelap dengan mengurangi intensitas pixel
dark = np.uint8(np.clip((0.5*(gray - min_val)), 0, 255))

# membuat tipe citra terang dengan menambahkan intensitas pixel
bright = np.uint8(np.clip((2*(gray - min_val)), 0, 255))

# membuat tipe citra kekontrasan rendah dengan mengecilkan rentang nilai pixel
low_contrast = cv2.normalize(gray, None, alpha=50, beta=150, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# membuat tipe citra kekontrasan tinggi dengan memperbesar rentang nilai pixel
high_contrast = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# membuat histogram untuk masing-masing tipe citra
hist_dark = cv2.calcHist([dark], [0], None, [256], [0, 256])
hist_bright = cv2.calcHist([bright], [0], None, [256], [0, 256])
hist_low_contrast = cv2.calcHist([low_contrast], [0], None, [256], [0, 256])
hist_high_contrast = cv2.calcHist([high_contrast], [0], None, [256], [0, 256])

# menampilkan gambar dan histogram
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0, 0].imshow(dark, cmap='gray')
axs[0, 0].set_title('Dark Image')
axs[0, 0].axis('off')
axs[0, 1].imshow(bright, cmap='gray')
axs[0, 1].set_title('Bright Image')
axs[0, 1].axis('off')
axs[1, 0].imshow(low_contrast, cmap='gray')
axs[1, 0].set_title('Low Contrast Image')
axs[1, 0].axis('off')
axs[1, 1].imshow(high_contrast, cmap='gray')
axs[1, 1].set_title('High Contrast Image')
axs[1, 1].axis('off')

fig2, axs2 = plt.subplots(2, 2, figsize=(10,10))
axs2[0, 0].plot(hist_dark)
axs2[0, 0].set_title('Histogram of Dark Image')
axs2[0, 1].plot(hist_bright)
axs2[0, 1].set_title('Histogram of Bright Image')
axs2[1, 0].plot(hist_low_contrast)
axs2[1, 0].set_title('Histogram of Low Contrast Image')
axs2[1, 1].plot(hist_high_contrast)
axs2[1, 1].set_title('Histogram of High Contrast Image')

plt.show()