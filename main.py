import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract


def gaussian_high_pass_filter(image, rad):
    # apply fourier transform
    F = np.fft.fftshift(np.fft.fft2(image))
    F = np.fft.fftshift(F)
    im_shape = F.shape
    # apply gaussian filter to get G
    H = np.zeros(im_shape)
    x, y = np.ogrid[:im_shape[0], :im_shape[1]]
    H[:, :] = 1 - np.exp(-(np.sqrt((x - im_shape[0] / 2) ** 2 + (y - im_shape[1] / 2) ** 2)) / (2 * rad ** 2))
    G = H * F
    # apply inverse fourier transform
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(G))
    reconstructed_image = np.abs(reconstructed_image)
    return reconstructed_image


def scale_to_255(img):
    result = np.absolute(img)
    (minVal, maxVal) = (np.min(result), np.max(result))
    result = (255 * ((result - minVal) / (maxVal - minVal))).astype("uint8")
    return result


def get_plate(img_path="./images/Cars2.png"):
    # load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur to remove noise
    blur = cv2.medianBlur(gray, 5)
    # get boundary using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(blur, kernel, iterations=2)
    boundary = blur - erode
    # apply gaussian high pass filter to sharpen the edges
    high_pass_img = gaussian_high_pass_filter(boundary, 30)
    high_pass_img = scale_to_255(high_pass_img)
    # apply otsu thresholding to get binary image
    edges = cv2.threshold(high_pass_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # define empty images to draw contours
    img_all_contours = np.ones_like(gray).astype(np.uint8) * 255
    img_k_contours = np.ones_like(gray).astype(np.uint8) * 255
    # find contours
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    cv2.drawContours(img_all_contours, contours[0], -1, (0, 255, 0), 1)
    # filter contours based on area
    contours = imutils.grab_contours(contours)
    # get first k candidate contours that might be the plate
    k = 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:k]
    # draw first k contours
    for x in contours:
        cv2.drawContours(img_k_contours, [x], 0, (0, 255, 0), 1)
    # get first probable plate
    plate = None
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w/h
        area = w*h
        # print(aspect_ratio, area)
        if 2.5 <= aspect_ratio <= 5 and 2000 <= area <= 120000:
            plate = c
            break

    crop_img = None
    if plate is not None:
        x, y, w, h = cv2.boundingRect(plate)
        crop_img = gray[y:y + h, x:x + w]
        text = get_plate_characters(crop_img)
        print(text)

    # display results
    fig, ax = plt.subplots(1, 8, figsize=(10, 5))
    ax[0].imshow(gray, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(blur, cmap='gray')
    ax[1].set_title('blur')
    ax[2].imshow(boundary, cmap='gray')
    ax[2].set_title('boundary')
    ax[3].imshow(high_pass_img, cmap='gray')
    ax[3].set_title('high_pass_img')
    ax[4].imshow(edges, cmap='gray')
    ax[4].set_title('edges')
    ax[5].imshow(img_all_contours, cmap='gray')  # 120 80 160 40
    ax[5].set_title('all contours')
    ax[6].imshow(img_k_contours, cmap='gray')  # 120 80 160 40
    ax[6].set_title('first k contours')
    if crop_img is not None:
        ax[7].imshow(crop_img, cmap='gray')  # 120 80 160 40
        ax[7].set_title('crop')


def get_plate_characters(plate_img):
    # to display results
    fig, ax1 = plt.subplots(1, 12, figsize=(10, 5))
    # preprocess the plate to separate characters, each character will be processed separately
    plate_img = gaussian_high_pass_filter(plate_img, 5)
    plate_img = scale_to_255(plate_img)
    plate_img = np.power(plate_img / 255.0, 1.1)
    plate_img = np.uint8(plate_img * 255)
    plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # show the plate after processing
    ax1[0].imshow(plate_img, cmap='gray')
    ax1[0].set_title('plate')
    # find contours
    contours = cv2.findContours(plate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    chars_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / h
        area = w * h
        # print(aspect_ratio, area)
        if .4 <= aspect_ratio <= .8 and 70 <= area <= 200:
            chars_contours.append([c, x])

    chars_contours = sorted(chars_contours, key=lambda x: x[1])
    plate_text = ''
    for i, (c, _) in enumerate(chars_contours):
        (x, y, w, h) = cv2.boundingRect(c)
        char = 255 - plate_img[y:y + h, x:x + w]
        # pad the character
        height, width = char.shape[:2]
        padd_sz = 2
        padded_char = np.ones((height + 2 * padd_sz, width + 2 * padd_sz), dtype=np.uint8) * 255
        padded_char[padd_sz:height + padd_sz, padd_sz:width + padd_sz] = char
        char = padded_char
        # enhance the character
        char = cv2.GaussianBlur(char, (3, 3), 0)
        char = cv2.threshold(char, 150, 255, cv2.THRESH_BINARY)[1]
        # use tesseract to get the character
        char_text = pytesseract.image_to_string(char, config=r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz').strip()
        plate_text += char_text
        ax1[i+1].imshow(char, cmap='gray')
        ax1[i+1].set_title(char_text)
    return plate_text


if __name__ == '__main__':
    get_plate()
    plt.show()
