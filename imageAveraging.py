import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar
img = cv2.imread('bloodCells.jpg', cv2.IMREAD_GRAYSCALE)

# menambahkan Gaussian noise aditif dengan mean 0 dan standar deviasi 64
noise = np.random.normal(0, 64, size=img.shape)
noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

# menentukan jumlah citra untuk di-averaging
K = [8, 16, 64, 128]

# melakukan image averaging untuk setiap K citra
averaged_images = []
for k in K:
    images = []
    for i in range(k):
        # menambahkan Gaussian noise aditif dengan mean 0 dan standar deviasi 64 ke gambar asli
        noise = np.random.normal(0, 64, size=img.shape)
        noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        images.append(noisy_img)

    # melakukan averaging untuk K citra
    averaged_img = np.mean(images, axis=0).astype(np.uint8)
    averaged_images.append(averaged_img)

# menampilkan gambar asli dan hasil averaging
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

for i in range(len(K)):
    k = K[i]
    averaged_img = averaged_images[i]
    axs[0, i + 1].imshow(averaged_img, cmap='gray')
    axs[0, i + 1].set_title('Average {} image'.format(k))
    axs[0, i + 1].axis('off')

    # menghitung perbedaan antara gambar asli dan hasil averaging
    diff = cv2.absdiff(img, averaged_img)
    axs[1, i + 1].imshow(diff, cmap='gray')
    axs[1, i + 1].set_title('Difference')
    axs[1, i + 1].axis('off')

plt.tight_layout()
plt.show()
