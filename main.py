# Author - Ranjith Dhanaraj.

# Import libraries.
import cv2
import numpy as np

# Defining main function.
def main():
    # Reading the original image.
    image_name = "1.jpg"
    src_image = cv2.imread(image_name)
    # Creating a HSV Image.
    hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    # Creating a greyscale image.
    grey_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    # Normalizing the image.
    normalized_image = cv2.normalize(grey_image, None, 0, 255, cv2.NORM_MINMAX)
    # Morphological operations.
    # kernel determines the intensity of the operations.
    kernel = np.ones((10, 10), np.uint8)
    # Opening removes stray noise from the image.
    opening_morph_image = cv2.morphologyEx(normalized_image, cv2.MORPH_OPEN, kernel)
    # Dilation increases the size of the blobs.
    dilate_morph_image = cv2.dilate(opening_morph_image, kernel, iterations = 1)
    
    # Finding contours.
    contours, _ = cv2.findContours(dilate_morph_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Drawing the bouding area and extracting HSV values.
    hsv_values = [None for i in range(len(contours))]
    # Cycling through all the contours.
    for i, c in enumerate(contours):
        # Obtaining the x, y co-ordinates along with width and height.
        x, y, w, h = cv2.boundingRect(c)
        image_with_boxes = (cv2.rectangle(src_image, (x, y), (x + w, y + h), 
                            (0, 255, 0), 2))
                            
        # Initializing a variable to store list of HSV values of the pixels 
        # in the mid point of the contour.
        hsv_contour = [None for i in range(0, h)]
        
        # Traversing through the y axis of the image with the help of height 
        # variable and the y co-ordinate of the contour.
        for idx in range(h):
            # Extracting the HSV values of the contour.
            hsv_contour[idx] = hsv_image[y + idx][int(x + (w / 2))]
            # Only use the following line for verification.
            # src_image[y + idx][int(x + (w / 2))] = [255, 255, 255]
        hsv_values[i] = hsv_contour
    
    output_image = image_with_boxes
    
    # Adding the number of contours to the image.
    output_image = (cv2.putText(output_image, 'Number of Contours Detected is ' 
                    + str(len(contours)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA))
    
    # Displaying the image.
    cv2.imshow("Output", output_image)
    # Debug line, not necessary.
    print(len(hsv_values[2]))
    
# Checking if this is the main execution file. If true then execute the main 
# function else exit.
if __name__ == "__main__":
    main()