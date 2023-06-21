import os, sys, imghdr, random
import cv2, imutils
from skimage import io
from skimage import transform as tf
import numpy as np
from datetime import datetime


def help() -> None:
    print("help:\n\tAugmentation.py [path_to_img]")


def augment(img_path: str) -> None:
    # img transformations with openCV
    img = cv2.imread(img_path)
    # flip
    flipped_img = cv2.flip(img, 1)
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Flip.JPG", flipped_img)
    #rotate
    rotated_img = imutils.rotate_bound(img, random.choice([-30, -20, -10, 10, 20, 30]))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Rotate.JPG", rotated_img)
    # contrast
    inc_contrast_img = cv2.convertScaleAbs(img, alpha=random.choice([1.1, 1.2, 1.3]), beta=1)
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Contrast.JPG", inc_contrast_img)
    # brightness
    inc_brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=random.choice([3, 5, 7]))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Brightness.JPG", inc_brightness_img)
    # shear
    num_rows, num_cols = img.shape[:2]
    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1]])
    dst_points = np.float32([[0,0], [int(random.choice([0.7, 0.6, 0.5])*(num_cols-1)),0], [int(random.choice([0.6, 0.5])*(num_cols-1)),num_rows-1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    img_shear = cv2.warpAffine(img, matrix, (num_cols,num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Shear.JPG", img_shear)
    # project
    src_points = np.float32([[0,0], [num_cols-1,0], [0,num_rows-1], [num_cols-1,num_rows-1]])
    dst_points = np.float32([[int(random.choice([0, 0.1, 0.2, 0.3])*num_cols),0], [int(random.choice([1.0, 0.9, 0.8, 0.7])*num_cols)-1,0], [int(random.choice([0, 0.1, 0.2, 0.3])*num_cols),num_rows-1], [int(random.choice([1.0, 0.9, 0.8, 0.7])*num_cols),num_rows-1]])
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_projection = cv2.warpPerspective(img, projective_matrix, (num_cols,num_rows))
    cv2.imwrite(img_path[0:len(img_path) - 4]+"_Projection.JPG", img_projection)


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
