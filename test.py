
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the image
image = cv2.imread('img/test_down/image_1_3_2.png')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format for matplotlib
plt.imshow(image_rgb)
plt.show()

# if is_file:
    # img = cv2.imread(file)
# else:
    # img = file

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
channels = cv2.split(image_rgb)
ch_str = ["r","g","b"]

for i, ch in enumerate(channels):
    cax = plt.imshow(ch)
    plt.colorbar(cax)
    plt.title(f"ch:{ch_str[i]}")
    plt.show()
