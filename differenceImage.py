import cv2
import numpy as np
import matplotlib.pyplot as plt

# mengimpor dua gambar
img1 = cv2.imread('bloodCells.jpg')
img2 = cv2.imread('bloodCells2.jpg')

# mengubah gambar menjadi grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# menghitung selisih absolut antara dua gambar
diff = cv2.absdiff(gray1, gray2)

# menerapkan thresholding pada gambar selisih
thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

# menerapkan operasi morfologi "opening" untuk menghilangkan noise
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# menampilkan hasilnya
fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs[0, 0].imshow(img1[:,:,::-1])
axs[0, 0].set_title('Image 1')

axs[0, 1].imshow(img2[:,:,::-1])
axs[0, 1].set_title('Image 2')

axs[1, 0].imshow(diff, cmap='gray')
axs[1, 0].set_title('Difference Image')

axs[1, 1].imshow(opening, cmap='gray')
axs[1, 1].set_title('Thresholded and Morphologically Opened')

plt.show()
