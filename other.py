import cv2
import numpy as np

def gabor_filter(image, theta=0, frequency=0.6, sigma_x=1, sigma_y=1):
    kernel = cv2.getGaborKernel((21, 21), sigma_x, sigma_y, frequency, 0.5, theta)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered_image

def extract_fingerprint_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_features = []
    for theta in orientations:
        filtered_image = gabor_filter(image, theta=theta)
        gabor_features.append(filtered_image)
    return gabor_features, image

def find_minutiae(image):
    _, binarized = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thinning = cv2.ximgproc.thinning(binarized)
    skeleton = thinning.copy()
    skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, _ = cv2.findContours(thinning, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours in red
    cv2.drawContours(skeleton, contours, -1, (0, 0, 255), 1)

    return skeleton

def display_images(images, titles):
    for i, (image, title) in enumerate(zip(images, titles)):
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'imagen\img1.PNG'
    gabor_features, original_image = extract_fingerprint_features(image_path)
    minutiae_image = find_minutiae(original_image)
    display_images(gabor_features + [minutiae_image], [f'Gabor Filtered Image {i+1}' for i in range(len(gabor_features))] + ['Minutiae'])

if __name__ == "__main__":
    main()
