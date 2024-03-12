import cv2
import numpy as np

def gabor_filter(image, theta=0, frequency=0.6, sigma_x=1, sigma_y=1):
    # Create Gabor kernel
    kernel = cv2.getGaborKernel((21, 21), sigma_x, sigma_y, frequency, 0.5, theta)

    # Apply Gabor filter to the image
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)

    return filtered_image

def extract_fingerprint_features(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gabor filter with different orientations
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []
    for theta in orientations:
        filtered_image = gabor_filter(image, theta=theta)
        gabor_features.append(filtered_image)

    return gabor_features

# Path to the input image
image_path = 'imagen\img1.PNG'

# Extract features
features = extract_fingerprint_features(image_path)

# Display the filtered images
for i, feature in enumerate(features):
    cv2.imshow(f'Gabor Filtered Image {i+1}', feature)

cv2.waitKey(0)
cv2.destroyAllWindows()

