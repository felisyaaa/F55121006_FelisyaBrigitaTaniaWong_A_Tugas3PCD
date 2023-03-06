import cv2
import numpy as np
from matplotlib import pyplot as plt

# memuat gambar asli
img = cv2.imread('bloodCells.jpg', cv2.IMREAD_GRAYSCALE)

# mengonversi gambar menjadi 8-bit (256 warna)
img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# mengekstrak empat plane bit urutan rendah
bit0 = img_8bit & 1
bit1 = img_8bit & 2
bit2 = img_8bit & 4
bit3 = img_8bit & 8

# menggabungkan keempat plane bit menjadi sebuah gambar baru
img_4bit = bit3 * 8 + bit2 * 4 + bit1 * 2 + bit0

# mengurangi empat plane bit urutan rendah dari gambar asli
diff = cv2.absdiff(img_8bit, img_4bit)

# melakukan histogram equalisasi pada gambar perbedaan
diff_eq = cv2.equalizeHist(diff)

# menampilkan gambar asli, gambar 4-bit, gambar perbedaan, dan gambar perbedaan yang disamakan histogramnya
plt.subplot(221)
plt.imshow(img_8bit, cmap='gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(img_4bit, cmap='gray')
plt.title('Image with 4 Lower Order Bit Planes Set to 0')

plt.subplot(223)
plt.imshow(diff, cmap='gray')
plt.title('Difference Image')

plt.subplot(224)
plt.imshow(diff_eq, cmap='gray')
plt.title('Histogram Equalized Difference Image')

plt.show()
