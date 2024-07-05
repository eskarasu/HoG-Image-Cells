import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# Resize the image (optional)
image = cv2.resize(image, (128, 64))

# Display the image
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# HoG parameters
cell_size = (8, 8)  # cell size
block_size = (2, 2)  # block size
nbins = 9  # number of histogram directions

# Calculate gradients
gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

# Calculate histograms based on cell sizes
cell_x, cell_y = cell_size
n_cellsx = image.shape[1] // cell_x  # number of cells in x axis
n_cellsy = image.shape[0] // cell_y  # number of cells in y axis

# Create an empty matrix for cell histograms
histogram = np.zeros((n_cellsy, n_cellsx, nbins))

# Fill the histogram
for i in range(n_cellsy):
    for j in range(n_cellsx):
        cell_magnitude = magnitude[i * cell_y:(i + 1) * cell_y, j * cell_x:(j + 1) * cell_x]
        cell_angle = angle[i * cell_y:(i + 1) * cell_y, j * cell_x:(j + 1) * cell_x]
        hist, _ = np.histogram(cell_angle, bins=nbins, range=(0, 180), weights=cell_magnitude)
        histogram[i, j, :] = hist

# Visualize the histograms
for i in range(n_cellsy):
    for j in range(n_cellsx):
        cell_hist = histogram[i, j, :]
        cell_hist /= cell_hist.sum()  # Normalize
        angles = np.linspace(0, 180, nbins, endpoint=False)
        for angle, mag in zip(angles, cell_hist):
            x_center = j * cell_x + cell_x // 2
            y_center = i * cell_y + cell_y // 2
            x1 = x_center + mag * cell_x // 2 * np.cos(np.deg2rad(angle))
            y1 = y_center - mag * cell_y // 2 * np.sin(np.deg2rad(angle))
            x2 = x_center - mag * cell_x // 2 * np.cos(np.deg2rad(angle))
            y2 = y_center + mag * cell_y // 2 * np.sin(np.deg2rad(angle))
            plt.plot([x1, x2], [y1, y2], 'r-')

plt.imshow(image, cmap='gray')
plt.title('HoG Cells')
plt.show()
