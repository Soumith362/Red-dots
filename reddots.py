 import cv2
import numpy as np
from matplotlib import pyplot as plt

def count_red_dots(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    
    # Step 2: Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Step 3: Define the red color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Step 4: Threshold the image to get only red colors
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

    # Step 5: Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Count dots
    dot_count = len(contours)
    
    # Optionally display the results
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 1)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title(f'Red Dots Counted: {dot_count}')
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return dot_count

# Example usage
image_path = r'C:\Users\Veladandi Soumith\OneDrive\Desktop\Customer.png'
number_of_dots = count_red_dots(image_path)
print(f'The number of red dots in the image is: {number_of_dots}')
