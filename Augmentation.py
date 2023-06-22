import os, sys, imghdr, random
import cv2, imutils
from skimage import io
from skimage import transform as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt



def help() -> None:
    print("help:\n\tAugmentation.py [path_to_img]")

def plotImage(img, img_change, title) :

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # Plot flipped image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_change, cv2.COLOR_BGR2RGB))
    plt.title(title)

    # Display the plot
    plt.show()


# FLIP

def flip(img_path, img) :

    flipped_img = cv2.flip(img, 1)
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Flip.JPG", flipped_img)

    #plot
    plotImage(img, flipped_img, 'Flipped Image')

def rotate(img_path, img) :

    #rotate
    rotated_img = imutils.rotate_bound(img, random.choice([-30, -20, -10, 10, 20, 30]))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Rotate.JPG", rotated_img)

    #plot
    plotImage(img, rotated_img, 'Rotated Image')

def contrast(img_path, img) :

    # contrast
    inc_contrast_img = cv2.convertScaleAbs(img, alpha=random.choice([1.1, 1.2, 1.3]), beta=1)
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Contrast.JPG", inc_contrast_img)

    #plot
    plotImage(img, inc_contrast_img, 'Darkest Image')

def brightness(img_path, img) :

    # brightness
    inc_brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=random.choice([3, 5, 7]))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Brightness.JPG", inc_brightness_img)

    #plot
    plotImage(img, inc_brightness_img, 'Brightness Image')

def shear(img_path, img) :

    # shear
    num_rows, num_cols = img.shape[:2]
    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1]])
    dst_points = np.float32([[0,0], [int(random.choice([0.7, 0.6, 0.5])*(num_cols-1)),0], [int(random.choice([0.6, 0.5])*(num_cols-1)),num_rows-1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    img_shear = cv2.warpAffine(img, matrix, (num_cols,num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Shear.JPG", img_shear)

    #plot
    plotImage(img, img_shear, 'Shear Image')

def projection(img_path, img) :

    # project
    num_rows, num_cols = img.shape[:2]
    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1], [num_cols-1,num_rows-1]])
    dst_points = np.float32([[int(random.choice([0, 0.1, 0.2, 0.3])*num_cols),0], [int(random.choice([1.0, 0.9, 0.8, 0.7])*num_cols)-1,0], [int(random.choice([0, 0.1, 0.2, 0.3])*num_cols),num_rows-1], [int(random.choice([1.0, 0.9, 0.8, 0.7])*num_cols),num_rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_projection = cv2.warpPerspective(img, projective_matrix, (num_cols,num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Projection.JPG", img_projection)

    #plot
    plotImage(img, img_projection, 'Projection Image')

def augment(img_path: str) -> None:
    # img transformations with openCV
    img = cv2.imread(img_path)

    #flip
    flip(img_path, img)

    #rotate
    rotate(img_path, img)

    # contrast
    contrast(img_path, img)

    # brightness
    brightness(img_path, img)

    # shear
    shear(img_path, img)

    # project
    projection(img_path, img)


def main() -> None:
    # set seed
    random.seed(datetime.now().timestamp())
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} does not exist".format(sys.argv[1]))
    if imghdr.what(sys.argv[1]) != 'jpeg':
        return print("Argument {} is not a jpeg img".format(sys.argv[1]))
    augment(sys.argv[1])


if __name__ == "__main__":
    main()
