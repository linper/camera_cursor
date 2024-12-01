
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the image
image = cv2.imread('img/ref_mixed.png')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format for matplotlib
plt.imshow(image_rgb)
plt.show()

# Step 2: Reshape the image data for clustering
# Reshape the image to be a list of pixels
pixels = image_rgb.reshape(-1, 3)

# Step 3: Apply K-means clustering to find the most common colors
num_colors = 7  # Number of colors in the palette
# num_colors = 1  # Number of colors in the palette
kmeans = KMeans(n_clusters=num_colors, random_state=0)
kmeans.fit(pixels)

# Get the cluster centers (the common colors)
palette = kmeans.cluster_centers_.astype(int)

# Step 4: Display the color palette using matplotlib
plt.figure(figsize=(8, 2))
plt.title('Color Palette')

# Create a bar for each color in the palette
for i, color in enumerate(palette):
    plt.bar(i, 1, color=np.array(color) / 255)  # Normalize RGB values to [0, 1] for matplotlib

# print(np.array(palette).tolist())
# print(palette[1:])
print(palette)

plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.show()
