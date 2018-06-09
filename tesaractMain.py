import cv2
import sys
import pytesseract


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def increase_img_contrast(img):
    b = 0.  # brightness
    c = 100.  # contrast

    # call addWeighted function, which performs:
    #    dst = src1*alpha + src2*beta + gamma
    # we use beta = 0 to effectively only operate on src1
    return cv2.addWeighted(img, 1. + c / 127., img, 0, b - c)


def get_text_from_image(img):
    # Uncomment the line below to provide path to tesseract manually
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' for using LSTM OCR Engine
    config = ('-l eng --oem 1 --psm 3')
    return pytesseract.image_to_string(img, config=config)


def show_image(img):
    cv2.imshow('Test', img)
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()


def analyse_image_by_path(path):
    img = read_img(path)
    img = increase_img_contrast(img)
    # Uncomment this if you want to show the image
    # show_image(img)
    print(get_text_from_image(img))


if __name__ == '__main__':
    analyse_image_by_path('img2.jpeg')

