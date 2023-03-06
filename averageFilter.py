import cv2
import numpy as np

# membaca gambar
img = cv2.imread('bloodCells.jpg')

# mengubah citra ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# membuat kernel filter rata-rata 3x3
kernel = np.ones((3, 3), np.float32) / 9

# melakukan operasi filter rata-rata
filtered = cv2.filter2D(gray, -1, kernel)

# menampilkan citra asli dan hasil filter rata-rata
cv2.imshow('Original Image', gray)
cv2.imshow('Average Filtered Image', filtered)

# menunggu tombol keyboard ditekan
cv2.waitKey(0)

# menutup semua jendela yang terbuka
cv2.destroyAllWindows()
