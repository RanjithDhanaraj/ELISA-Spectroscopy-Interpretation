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
    # Morphological operations
    kernel = np.ones((10, 10), np.uint8)
    opening_morph_image = cv2.morphologyEx(normalized_image, cv2.MORPH_OPEN, kernel)
    dilate_morph_image = cv2.dilate(opening_morph_image, kernel, iterations = 1)
    
    # Finding contours
    contours, _ = cv2.findContours(dilate_morph_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Drawing the bouding area
    image_with_boxes = cv2.drawContours(src_image, contours, -1, (255, 255, 255), 2)
    output_image = image_with_boxes
    # Adding the number of contours to the text
    output_image = (cv2.putText(output_image, 'Number of Contours Detected is ' 
                    + str(len(contours)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA))
    
    # Displaying the image 
    cv2.imshow("Output", output_image)
    
    
if __name__ == "__main__":
    main()