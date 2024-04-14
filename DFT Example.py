import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/Users/DAVID/Desktop/dft.jpg', cv2.IMREAD_GRAYSCALE)

# Perform FFT on the image
fft_image = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_image)

# Calculate amplitude and phase
amplitude = np.abs(fft_shifted)
phase = np.angle(fft_shifted)

# Display original image, DFT image, and reversed image
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(amplitude), cmap='gray')
plt.title('Amplitude Spectrum')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(phase, cmap='gray')
plt.title('Phase Spectrum')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.show()

# Reverse FFT to obtain the original image
reversed_fft = np.fft.ifftshift(fft_shifted)
reversed_image = np.fft.ifft2(reversed_fft)
reversed_image = np.abs(reversed_image).astype(np.uint8)

# Display original, DFT, and reversed images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(amplitude), cmap='gray')
plt.title('DFT Image')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reversed_image, cmap='gray')
plt.title('Reversed Image (Inverse DFT)')
plt.axis('off')

plt.tight_layout()
plt.show()