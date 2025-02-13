# Autor: Pedro Castello
# Date: 13 of February of 2025
# Description: This script is a test of the extraction of the channels of an image
# and the conversion of an image to grayscale using OpenCV and Matplotlib and the
# manipulation of images to train AI models.

import pandas as pd
import numpy as np
import os

from glob import glob
from PIL import Image

import cv2
import matplotlib.pylab as plt

def saveImages(images, saveDirectory):
    '''
    Saves a dictionary where each key corresponds to an
    image, storing them in a specified directory.

    Parameters:
    - images (type=dict): Dictionary where keys are the names of the images and values are the image data.
    - saveDirectory (type=str): path that the function will save the images. 
    '''
    print("Saving the treated images in {}..." .format(saveDirectory))
    os.makedirs(saveDirectory, exist_ok=True)

    errors = []

    for name, image in images.items():
        path = os.path.join(saveDirectory, f"{name}.png")
        try:
            if isinstance(image, plt.Figure):
                image.savefig(path, dpi=300)

            elif isinstance(image, np.ndarray):
                if image.shape[-1] == 3:  # If the image is colored
                    # Converts to BGR before saving to avoid color issues
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                if not cv2.imwrite(path, image):
                    raise ValueError("Failed to save with OpenCV.")
            
            elif isinstance(image, Image.Image):
                image.save(path)
            
            else:
                raise TypeError(f"O timpo de imagem de {name} não foi reconhecido")

        except Exception as e:
            errors.append(f"{name}: {str(e)}")

    if not errors:
        print(f"All images were saved successfully!")
    else:
        print("Some images could not be saved correctly:")
        for error in errors:
            print(f"    - {error}")

if __name__ == "__main__":
    print('Available styles in plt.style.use: ', plt.style.available)

    plt.style.use('ggplot')

    print('OpenCV version:', cv2.__version__)
    print('Numpy version:', np.__version__)
    print('Pandas version:', pd.__version__)
    print('Matplotlib version:', plt.__version__)

    # Loading images into variables and converting one of them to grayscale
    cat_files = glob('CatsAndDogsDataset/cats/*.jpg')
    dog_files = glob('CatsAndDogsDataset/dogs/*.jpg')
    imgCat_mpl = plt.imread(cat_files[27])
    imgCat_cv2 = cv2.imread(cat_files[27])
    imgCat_gray = cv2.cvtColor(imgCat_cv2, cv2.COLOR_BGR2GRAY)

    numPixels = imgCat_mpl.size // 3

    # pixelValues is a pandas series with the pixel values of the cat image
    # pd.Series(imgCat_mpl.flatten()) flattens the image and converts it to a pandas series
    # that means that pandas will transform the image into a 1D array from
    # a 3D array(Height, Width, Channels)
    pixelValues = pd.Series(imgCat_mpl.flatten())

    # First screen - Variations of color channels and grayscale of a cat image
    fig1, axs1 = plt.subplots(1, 5, figsize=(25, 5))

    axs1[0].imshow(imgCat_mpl)
    axs1[1].imshow(imgCat_mpl[:,:,0], cmap='Reds') 
    axs1[2].imshow(imgCat_mpl[:,:,1], cmap='Greens')
    axs1[3].imshow(imgCat_mpl[:,:,2], cmap='Blues')
    axs1[4].imshow(imgCat_gray, cmap='gray')

    axs1[0].axis('off')
    axs1[1].axis('off')
    axs1[2].axis('off')
    axs1[3].axis('off')
    axs1[4].axis('off')

    axs1[0].set_title('Original Image')
    axs1[1].set_title('Red channel')
    axs1[2].set_title('Green channel')
    axs1[3].set_title('Blue channel')
    axs1[4].set_title('Grayscale Image')
    plt.show()

    # Second screen - Difference between matplotlib and OpenCV
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    axs2[0].imshow(imgCat_mpl)
    axs2[0].set_title('Matplotlib Image')
    axs2[0].axis('off')

    axs2[1].imshow(imgCat_cv2)
    axs2[1].set_title('OpenCV Image (BGR)')
    axs2[1].axis('off')

    axs2[2].imshow(cv2.cvtColor(imgCat_cv2, cv2.COLOR_BGR2RGB))
    axs2[2].set_title('OpenCV Image (Transformed into RGB)')
    axs2[2].axis('off')

    # Third screen - Histograms
    fig2, axs3 = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram of RGB pixel values from the original image
    axs3[0].hist(imgCat_mpl.flatten(), bins=256, edgecolor='black', linewidth=0.1)
    axs3[0].set_title('Distribution of Pixel Values (total: {}px)'.format(numPixels))
    axs3[0].set_xlabel('Pixel Value')
    axs3[0].set_ylabel('Frequency')

    # Histogram of grayscale pixel values from the grayscale image
    axs3[1].hist(imgCat_gray.flatten(), bins=256, color='gray', edgecolor='white', linewidth=0.1)
    axs3[1].set_title('Histogram of Grayscale Pixel Values (total: {}px)'.format(numPixels))
    axs3[1].set_xlabel('Pixel Value')
    axs3[1].set_ylabel('Frequency')
    plt.show()

    # Forth screen - Image of a dog in grayscale, resized to an smaller image and bigger image and stretched image
    dogImg_mpl = plt.imread(dog_files[50])
    dogImgGray = cv2.cvtColor(dogImg_mpl, cv2.COLOR_BGR2GRAY)
    dogImg_resized2Small = cv2.resize(dogImg_mpl, None, fx=0.25, fy=0.25)
    dogImg_resized2Bigger = cv2.resize(dogImg_mpl, (5000, 5000), interpolation=cv2.INTER_CUBIC)
    dogImg_stretched = cv2.resize(dogImg_mpl, (100,200))
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(dogImg_mpl)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(dogImgGray, cmap='gray')
    ax[1].set_title('Grayscale Image')
    ax[1].axis('off')
    ax[2].imshow(dogImg_resized2Small)
    ax[2].set_title('Resized Smaller Image')
    ax[2].axis('off')
    ax[3].imshow(dogImg_resized2Bigger)
    ax[3].set_title('Resized Bigger Image')
    ax[3].axis('off')
    ax[4].imshow(dogImg_stretched)
    ax[4].set_title('Stretched Image')
    ax[4].axis('off')
    plt.show()

    # Fifth Screen - CV2 Kernels
    # Sharpen an image
    imgCat_kernel = plt.imread(cat_files[123])
    kernel_sharpening = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]) # This is a matrix 3x3
    imgCat_kernelSharpened = cv2.filter2D(imgCat_kernel, -1, kernel_sharpening) #(image, depth, kernel)
    # Blurring an image
    kernel_blurry = np.ones((3,3), np.float32) / 9 # (dimensions, type_of_data) / 9
    imgCat_kernelBlurry = cv2.filter2D(imgCat_kernel, -1, kernel_blurry)

    # Showing plots
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].imshow(imgCat_kernel)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(imgCat_kernelSharpened)
    ax[1].set_title('Sharpened Image')
    ax[1].axis('off')
    ax[2].imshow(imgCat_kernelBlurry)
    ax[2].set_title('Blurry Image')
    ax[2].axis('off')
    plt.show()

    # Printing the pixel values of the cat image in the console
    print('---------------------------------- Pixel Values Sample Start ------------------------------------')
    print(pixelValues)
    print('---------------------------------- Pixel Values Sample End ------------------------------------')

    # Printing the dimensions of the images in the console
    print('imgCat_mpl dimensions: ', imgCat_mpl.shape)
    print('imgCat_cv2 dimensions: ', imgCat_cv2.shape)
    print('imgCat_gray dimensions: ', imgCat_gray.shape)
    print('dogImg dimensions', dogImg_mpl.shape)
    print('dogImgGray dimensions: ', dogImgGray.shape)
    print("dogImg_resized2Small dimensions: [{}]" .format(dogImg_resized2Small.shape))
    print("dogImg_resized2Bigger dimensions: [{}]" .format(dogImg_resized2Bigger.shape))
    print("note that compared to the original image [{}], dogImdogImg_resized2Small is" .format(dogImg_mpl.shape))
    print("75% smaller and dogImg_resized2Bigger's size is 5000x5000")

    # Saving the treated images
    saveDirectory = "treatedImages"

    images = {
        "imgCat_mpl": imgCat_mpl,
        "imgCat_cv2": imgCat_cv2,
        "imgCat_gray": imgCat_gray,
        "imgCat_kernel": imgCat_kernel,
        "imgCat_kernelSharpened": imgCat_kernelSharpened,
        "imgCat_kernelBlurry": imgCat_kernelBlurry,
        "dogImg": dogImg_mpl,
        "dogImgGray": dogImgGray,
        "dogImg_resized2Small": dogImg_resized2Small,
        "dogImg_resized2Bigger": dogImg_resized2Bigger,
        "dogImg_stretched": dogImg_stretched
    }

    saveImages(images, saveDirectory)