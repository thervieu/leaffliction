import os, sys, filetype, random, fnmatch
from plantcv import plantcv as pcv
import numpy as np
import click
from datetime import datetime


def find(pattern, path):
    """
    Finds the file whose name matches the requested pattern
    Arguments:
        pattern (string): requested pattern
        path (string): path where the file must be searched
    Returns:
        Complete path to the file, or None if file wasn't found
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return None

def transform_image(img_path: str, dst: str) -> None:
    """
    Performs six image transformations on an image using PlantCV's functions
    Arguments:
        img_path (string): path to the image on which transformations are to be performed
        dst (string): directory where resulting images will be stored
    """
    
    # Initialisation
    img, path, filename = pcv.readimage(img_path)
    pcv.params.debug_outdir = dst
    if not os.path.exists(dst):
        os.makedirs(dst)
    new_image_prefix = dst + '/' + (img_path[2:len(img_path) - 4] if img_path[0:2] == "./" else img_path[0:len(img_path) - 4])
    new_image_directory = os.path.split(new_image_prefix)[0]
    if not os.path.exists(new_image_directory):
        os.makedirs(new_image_directory)

    # Generating the black and white mask necessary for many transformations
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255, object_type='dark')
    mask = pcv.invert(s_thresh)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)
    
    # Gaussian blur of mask
    gaussian_blur = pcv.gaussian_blur(mask, ksize=(3,3))
    pcv.print_image(gaussian_blur, new_image_prefix + "_PCVBLUR.JPG")
    
    # Masked image
    masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
    pcv.print_image(masked, new_image_prefix + "_MASKED.JPG")
    
    # Objects and contours
    contour, hierarchy = pcv.roi.rectangle(img, 0, 0, img.shape[0], img.shape[1])
    objects, object_hierarchy = pcv.find_objects(img, mask)
    pcv.params.debug = 'print'
    pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy, object_contour=objects, obj_hierarchy=object_hierarchy, roi_type='partial')
    generated_file = find("*_obj_on_img.png", dst)
    os.rename(generated_file, new_image_prefix + "_ROI_OBJECTS.JPG")
    remove_trash_file = find("*_roi_mask.png", dst)
    os.remove(remove_trash_file)
    pcv.params.debug = 'None'

    # Analyse object
    obj, mask = pcv.object_composition(img=img, contours=objects, hierarchy=object_hierarchy)
    analyse_img = pcv.analyze_object(img, obj, mask)
    pcv.print_image(analyse_img, new_image_prefix + "_ANALYZED.JPG")

    # Pseudolandmarks
    pcv.params.debug = 'print'
    pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
    generated_file = find("*_pseudolandmarks.png", dst)
    os.rename(generated_file, new_image_prefix + "_PSEUDOLANDMARKS.JPG")
    pcv.params.debug = 'None'

    # Analyse colors
    color_channels = pcv.analyze_color(rgb_img=img, mask=mask, colorspaces='all', label="default")
    pcv.print_image(color_channels, new_image_prefix + "_COLORS.JPG")


def transform_directory(src: str, dst: str) -> None:
    for filename in os.listdir(src):
        file_path = os.path.join(src, filename)
        if os.path.isfile(file_path):
            transform_image(img_path=file_path, dst=dst)
        elif os.path.isdir(file_path):
            transform_directory(src=file_path, dst=dst)


@click.command()
@click.option('--src', default=None, help='Directory on which to perform transformations')
@click.option('--dst', default=None, help="Storage directory of the transformed data")
@click.argument('file', required=False)
def main(file, src, dst) -> None:

    # Set random seed
    random.seed(datetime.now().timestamp())

    if file is not None:
        if os.path.isfile(file) is False:
            return print(f"{file} does not exist or is not a file")
        if (filetype.guess(file) == None or filetype.guess(file).extension != 'jpg'):
            return print(f"{file} is not a jpeg image")
        transform_image(file, dst='.')
    elif (src is not None and dst is not None):
        if os.path.isdir(src) is False:
            return print(f"{src} does not exist or is not a directory")
        transform_directory(src, dst)
    else:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print(f"You must either provide a [src] and [dst] directory, or a source file as arguments for the transformation")
        ctx.exit()



if __name__ == "__main__":
    main()
