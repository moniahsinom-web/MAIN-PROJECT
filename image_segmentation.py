# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image
image = cv2.imread('bird.jpg')   # replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Reshape image (convert to 2D array of pixels)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Step 3: Define criteria and number of clusters (K)
k = 3   # you can change this
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Step 4: Apply K-Means clustering
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Step 5: Convert centers back to uint8
centers = np.uint8(centers)

# Step 6: Map labels to cluster centers
segmented_image = centers[labels.flatten()]

# Step 7: Reshape back to original image shape
segmented_image = segmented_image.reshape(image.shape)

# Step 8: Show images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis('off')

plt.show()