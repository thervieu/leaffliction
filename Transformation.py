import os, sys, filetype, random, fnmatch
from plantcv import plantcv as pcv
import numpy as np
from datetime import datetime


def usage():
    print("usage:\n\tTransformation.py [path_to_img or path_to_directory]")


def find(pattern, path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None

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
    objects, object_hierarchy = pcv.find_objects(img, mask)
    pcv.params.debug = 'print'
    pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy, object_contour=objects, obj_hierarchy=object_hierarchy, roi_type='partial')
    generated_file = find("*_obj_on_img.png", pcv.params.debug_outdir)
    os.rename(generated_file, img_path[0:len(img_path) - 4]+"_ROI_OBJECTS.JPG")
    remove_trash_file = find("*_roi_mask.png", pcv.params.debug_outdir)
    os.remove(remove_trash_file)
    pcv.params.debug = 'None'

    # Analyze object
    obj, mask = pcv.object_composition(img=img, contours=objects, hierarchy=object_hierarchy)
    analyse_img = pcv.analyze_object(img, obj, mask)
    pcv.print_image(analyse_img, img_path[0:len(img_path) - 4]+"_ANALYZED.JPG")

    # Pseudolandmarks
    pcv.params.debug = 'print'
    pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
    generated_file = find("*_pseudolandmarks.png", pcv.params.debug_outdir)
    os.rename(generated_file, img_path[0:len(img_path) - 4]+"_PSEUDOLANDMARKS.JPG")
    pcv.params.debug = 'None'

    # Analyse colors
    color_channels = pcv.analyze_color(rgb_img=img, mask=mask, hist_plot_type='all', label="default")
    pcv.print_image(color_channels, img_path[0:len(img_path) - 4]+"_COLORS.JPG")


def main() -> None:
    # set seed
    random.seed(datetime.now().timestamp())
    # argument
    if len(sys.argv) != 2:
        return usage()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} does not exist".format(sys.argv[1]))
    if (filetype.guess(sys.argv[1]) == None or (filetype.guess(sys.argv[1])).extension != 'jpg'):
        return print("Argument {} is not a jpeg img".format(sys.argv[1]))
    transform(sys.argv[1])


if __name__ == "__main__":
    main()
