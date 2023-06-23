import os, sys, imghdr, random
from plantcv import plantcv as pcv
import numpy as np
from datetime import datetime


def usage():
    print("usage:\n\tTransformation.py [path_to_img or path_to_directory]")


def transform(img_path: str) -> None:
    # img transformations with openCV
    img, path, filename = pcv.readimage(img_path)
    pcv.params.debug_outdir = os.path.split(img_path)[0]
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255, object_type='dark')
    mask = pcv.invert(s_thresh)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    # gaussian blur
    gaussian_blur = pcv.gaussian_blur(mask, ksize=(3,3))
    pcv.print_image(gaussian_blur, img_path[0:len(img_path) - 4]+"_PCVBLUR.JPG")
    # mask
    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    pcv.print_image(masked, img_path[0:len(img_path) - 4]+"_MASKED.JPG")
    # Objects and contours
    contour, hierarchy = pcv.roi.rectangle(img, 0, 0, img.shape[0], img.shape[1])
    pcv.params.debug = 'print'
    objects, object_hierarchy = pcv.find_objects(img, mask)
    pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy, object_contour=objects, obj_hierarchy=object_hierarchy, roi_type='partial')
    # Analyze object
    obj, mask = pcv.object_composition(img=img, contours=objects, hierarchy=object_hierarchy)
    analyse_img = pcv.analyze_object(img, obj, mask)
    pcv.print_image(analyse_img, img_path[0:len(img_path) - 4]+"_ANALYZED.JPG")
    pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
    pcv.params.debug = 'None'

def main() -> None:
    # set seed
    random.seed(datetime.now().timestamp())
    # argument
    if len(sys.argv) != 2:
        return usage()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} does not exist".format(sys.argv[1]))
    if imghdr.what(sys.argv[1]) != 'jpeg':
        return print("Argument {} is not a jpeg img".format(sys.argv[1]))
    transform(sys.argv[1])


if __name__ == "__main__":
    main()
