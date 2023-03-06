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

# equalisasi histogram untuk masing-masing tipe citra
eq_dark = cv2.equalizeHist(dark)
eq_bright = cv2.equalizeHist(bright)
eq_low_contrast = cv2.equalizeHist(low_contrast)
eq_high_contrast = cv2.equalizeHist(high_contrast)

# membuat histogram untuk masing-masing citra yang telah di-equalisasikan
eq_hist_dark = cv2.calcHist([eq_dark], [0], None, [256], [0, 256])
eq_hist_bright = cv2.calcHist([eq_bright], [0], None, [256], [0, 256])
eq_hist_low_contrast = cv2.calcHist([eq_low_contrast], [0], None, [256], [0, 256])
eq_hist_high_contrast = cv2.calcHist([eq_high_contrast], [0], None, [256], [0, 256])

# menampilkan gambar dan histogram sebelum dan sesudah equalisasi
fig, axs = plt.subplots(4, 3, figsize=(15, 10))
axs[0, 0].imshow(dark, cmap='gray')
axs[0, 0].set_title('Dark Image Before Equalization')
axs[0, 0].axis('off')
axs[0, 1].imshow(eq_dark, cmap='gray')
axs[0, 1].set_title('Dark Image After Equalization')
axs[0, 1].axis('off')
axs[0, 2].plot(eq_hist_dark)
axs[0, 2].set_title('Equalized Histogram of Dark Image')

axs[1, 0].imshow(bright, cmap='gray')
axs[1, 0].set_title('Bright Image Before Equalization')
axs[1, 0].axis('off')
axs[1, 1].imshow(eq_bright, cmap='gray')
axs[1, 1].set_title('Bright Image After Equalization')
axs[1, 1].axis('off')
axs[1, 2].plot(eq_hist_bright)
axs[1, 2].set_title('Equalized Histogram of Bright Image ')

axs[2, 0].imshow(low_contrast, cmap='gray')
axs[2, 0].set_title('Low Contrast Image Before Equalization')
axs[2, 0].axis('off')
axs[2, 1].imshow(eq_low_contrast, cmap='gray')
axs[2, 1].set_title('Low Contrast Image After Equalization')
axs[2, 1].axis('off')
axs[2, 2].plot(eq_hist_low_contrast)
axs[2, 2].set_title('Equalized Histogram of Low Contrast Image')

axs[3, 0].imshow(high_contrast, cmap='gray')
axs[3, 0].set_title('High Contrast Image Before Equalization')
axs[3, 0].axis('off')
axs[3, 1].imshow(eq_high_contrast, cmap='gray')
axs[3, 1].set_title('High Contrast Image After Equalization')
axs[3, 1].axis('off')
axs[3, 2].plot(eq_hist_high_contrast)
axs[3, 2].set_title('Equalized Histogram of High Contrast Image')

fig.tight_layout()
plt.show()