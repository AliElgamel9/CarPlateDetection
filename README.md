# CarPlateDetection
This repository contains an image processing solution for detecting and classifying car license plates. The goal of this project is to develop a robust system that can accurately identify and analyze the alphanumeric characters present on a car license plate, using various image processing techniques.

The proposed methodology:
1- Prepare and load the image

2- Preprocessing: The preprocessed image is converted to grayscale and further processed to remove noise using median blur. Morphological operations are applied to obtain the boundary of the license plate region.

3- Gaussian High Pass Filter: The image is transformed using Fourier transform and then passed through a Gaussian high pass filter to sharpen the edges of the license plate.

4-Otsu Thresholding: Otsu thresholding is applied to convert the high pass filtered image into a binary image, where the license plate region is highlighted.

5-Contour Detection: Contours are detected on the binary image using the findContours function. All the contours are drawn on a separate image for visualization purposes.

6-Contour Filtering: The contours are filtered based on their area and aspect ratio to extract potential license plate candidates. The top-k (where k is a predefined value) contours with the largest areas are selected as potential license plate regions.

7-Character Segmentation: For each potential license plate candidate, the characters are segmented by applying further processing steps, including another Gaussian high pass filter, scaling, power transformation, and thresholding.

8-Character Recognition: Contours are detected on the processed plate image, and the characters are extracted based on their aspect ratio and area. Each character is further enhanced, padded, and processed to improve readability. Tesseract, an optical character recognition (OCR) engine, is then used to recognize the characters, which are concatenated to form the final license plate text.

9-Visualization: The results are displayed using matplotlib, with multiple subplots showing the intermediate and final images at various stages of the process.

In summary, the methodology combines various image processing techniques, including filtering, thresholding, contour analysis, and character segmentation, along with OCR for character recognition, to achieve license plate detection and classification.
