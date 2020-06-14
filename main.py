# Code to extract 
import cv2
import numpy as np

def main():
    # Reading the original image
    image_name = "1.jpg"
    src_image = cv2.imread(image_name)
    # Creating a greyscale image
    grey_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # Normalizing the image
    normalized_image = cv2.normalize(grey_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Finding contours
    contours, _ = cv2.findContours(normalized_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Drawing the bouding area
    image_with_boxes = cv2.drawContours(src_image, contours, -1, (255, 255, 255), 2)
    output_image = image_with_boxes
    # Displaying the image 
    cv2.imshow("Output", output_image)
    
if __name__ == "__main__":
    main()