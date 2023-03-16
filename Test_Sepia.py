import numpy as np
import cv2

def sepia(src_image):
    # convert the input image to grayscale
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    
    # normalize the grayscale values to the range of 0 to 1
    normalized_gray = np.array(gray, np.float32)/255
    
    # create a 3-channel array with solid sepia color
    sepia = np.ones(src_image.shape)
    sepia[:,:,0] *= 153 # B
    sepia[:,:,1] *= 204 # G
    sepia[:,:,2] *= 255 # R

    print(sepia[:,:,0])
    # apply the weighting factors to the sepia color channels using element-wise multiplication
    sepia[:,:,0] *= normalized_gray # B
    sepia[:,:,1] *= normalized_gray # G
    sepia[:,:,2] *= normalized_gray # R
    
    # convert the resulting sepia image to uint8 data type
    sepia_img = np.array(sepia, dtype=np.uint8)
   
    return sepia_img

def invert_method(img):
    img_gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(img_gray)

    return invert_img
# read an input image
image = cv2.imread("E:\Git-hab\Image_Processing_GUI\ohm1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Invert image",invert_method(image))
# apply the sepia filter to the input image
sepia_image = sepia(image)

# display the resulting sepia image
cv2.imshow("Gray Image", gray)

# wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()