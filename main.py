import cv2

# Load the image
img = cv2.imread('img1.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray, 5)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh, None, iterations=15)
thresh = cv2.erode(thresh, None, iterations=15)

# Find the contours
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Finally show the image
cv2.imshow('img', img)
cv2.imshow('res', thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
