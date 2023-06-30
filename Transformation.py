import os
import filetype
import fnmatch
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import click


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


def transform_image(img_path: str, dst: str, type: str) -> None:
    """
    Performs six image transformations on an image using PlantCV's functions
    Arguments:
        img_path (string): path to original image
        dst (string): directory where resulting images will be stored
        type (string): type of transformation requested
    """

    # Initialisation
    img, path, filename = pcv.readimage(img_path)
    pcv.params.debug_outdir = dst
    if not os.path.exists(dst):
        os.makedirs(dst)
    new_image_prefix = dst + '/' + (img_path[2:len(img_path) - 4]
                                    if img_path[0:2] == "./"
                                    else img_path[0:len(img_path) - 4])
    new_image_directory = os.path.split(new_image_prefix)[0]
    if not os.path.exists(new_image_directory):
        os.makedirs(new_image_directory)

    # Generating the black and white mask necessary for many transformations
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=60, max_value=255,
                                    object_type='dark')
    mask = pcv.invert(s_thresh)
    mask = pcv.erode(gray_img=mask, ksize=3, i=1)

    # Gaussian blur of mask
    if type in ['blur', 'all']:
        gaussian_blur = pcv.gaussian_blur(mask, ksize=(3, 3))
        pcv.print_image(gaussian_blur, new_image_prefix + "_PCVBLUR.JPG")

    # Masked image
    if type in ['mask', 'all']:
        masked = pcv.apply_mask(img=img, mask=mask, mask_color='white')
        pcv.print_image(masked, new_image_prefix + "_MASKED.JPG")

    # Common part for next transformations
    if type in ['roi', 'all', 'analysis', 'pseudolandmarks']:
        objects, object_hierarchy = pcv.find_objects(img, mask)

    # ROI objects
    if type in ['roi', 'all']:
        contour, hierarchy = pcv.roi.rectangle(img, 0, 0, img.shape[0],
                                               img.shape[1])
        pcv.params.debug = 'print'
        pcv.roi_objects(img=img, roi_contour=contour, roi_hierarchy=hierarchy,
                        object_contour=objects, obj_hierarchy=object_hierarchy,
                        roi_type='partial')
        generated_file = find("*_obj_on_img.png", dst)
        os.rename(generated_file, new_image_prefix + "_ROI_OBJECTS.JPG")
        remove_trash_file = find("*_roi_mask.png", dst)
        os.remove(remove_trash_file)
        pcv.params.debug = 'None'

    # Common part for next transformations
    if type in ['analysis', 'all', 'pseudolandmarks']:
        obj, mask = pcv.object_composition(img=img, contours=objects,
                                           hierarchy=object_hierarchy)

    # Analyse objects
    if type in ['analysis', 'all']:
        analyse_img = pcv.analyze_object(img, obj, mask)
        pcv.print_image(analyse_img, new_image_prefix + "_ANALYZED.JPG")

    # Pseudolandmarks
    if type in ['pseudolandmarks', 'all']:
        pcv.params.debug = 'print'
        pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask,
                                   label="default")
        generated_file = find("*_pseudolandmarks.png", dst)
        os.rename(generated_file, new_image_prefix + "_PSEUDOLANDMARKS.JPG")
        pcv.params.debug = 'None'

    # Analyse colors channels
    if type in ['colors', 'all']:
        color_channels = pcv.analyze_color(rgb_img=img, mask=mask,
                                           colorspaces='all', label="default")
        pcv.print_image(color_channels, new_image_prefix + "_COLORS.JPG")


def transform_directory(src: str, dst: str, type: str) -> None:
    """
    Performs image transformations on every image of a directory,
    including images in sub-directories
    Arguments:
        src (string): path of directory where transformations will be applied
        dst (string): directory where resulting images will be stored
        type (string): type of transformation requested
    """
    file_count = 0
    for filename in os.listdir(src):
        file_path = os.path.join(src, filename)
        if os.path.isfile(file_path):
            transform_image(img_path=file_path, dst=dst, type=type)
            file_count += 1
        elif os.path.isdir(file_path):
            transform_directory(src=file_path, dst=dst, type=type)
    print(f"Applied {type} transformations to {file_count} files in {src}")


def plot_images(img_path: str, dst: str) -> None:
    """
    In the case where only one file is requested, assignment requires to
    display the set of image transformations. This does it
    Arguments:
        img_path (str): Path to original image
        dst (str): Destination where the transformations were stored
    """

    # Get the paths to each transformed image
    img_prefix = dst + '/' + (img_path[2:len(img_path) - 4]
                              if img_path[0:2] == "./"
                              else img_path[0:len(img_path) - 4])
    blur_path = img_prefix + "_PCVBLUR.JPG"
    mask_path = img_prefix + "_MASKED.JPG"
    roi_path = img_prefix + "_ROI_OBJECTS.JPG"
    analysis_path = img_prefix + "_ANALYZED.JPG"
    landmark_path = img_prefix + "_PSEUDOLANDMARKS.JPG"
    colors_path = img_prefix + "_COLORS.JPG"

    # Check existence of transformed files
    image_names = ['Original Image']
    img_blur, img_mask, img_roi, img_analyzed, img_landmarks, img_colors = (None,)*6
    if os.path.isfile(blur_path):
        img_blur, path, filename = pcv.readimage(blur_path)
        image_names.append('Blurred Mask')
    if os.path.isfile(mask_path):
        img_mask, path, filename = pcv.readimage(mask_path)
        image_names.append('Masked Image')
    if os.path.isfile(roi_path):
        img_roi, path, filename = pcv.readimage(roi_path)
        image_names.append('ROI Objects')
    if os.path.isfile(analysis_path):
        img_analyzed, path, filename = pcv.readimage(analysis_path)
        image_names.append('Analysed Objects')
    if os.path.isfile(landmark_path):
        img_landmarks, path, filename = pcv.readimage(landmark_path)
        image_names.append('Pseudolandmarks')
    if os.path.isfile(colors_path):
        img_colors, path, filename = pcv.readimage(colors_path)

    # Initialise plot
    length = len(image_names)
    rows = 1 if length <= 3 else 2
    cols = length if length <= 3 else 3
    plotted = 1

    # Plot original image
    img_original, path, filename = pcv.readimage(img_path)
    plt.subplot(rows, cols, plotted)
    plt.imshow(img_original)
    plt.title('Original Image')
    plotted += 1

    # Plot blurred image
    if img_blur is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_blur)
        plt.title("Blurred mask")
        plotted += 1

    # Plot masked image
    if img_mask is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_mask)
        plt.title("Masked image")
        plotted += 1

    # Plot roi image
    if img_roi is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_roi)
        plt.title("ROI objects")
        plotted += 1

    # Plot analyzed image
    if img_analyzed is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_analyzed)
        plt.title("Analyzed objects")
        plotted += 1

    # Plot image's landmarks
    if img_landmarks is not None:
        plt.subplot(rows, cols, plotted)
        plt.imshow(img_landmarks)
        plt.title("Pseudolandmarks")
        plotted += 1

    # Display 5 first transformations with original image
    plt.show()

    # Display color analysis of image afterwards
    if img_colors is not None:
        plt.imshow(img_colors)
        plt.title("Color histogram")
        plt.show()


@click.command()
@click.option('--src', default=None, help='Directory of original data')
@click.option('--dst', default=None, help="Directory of the transformed data")
@click.option('--type', default='all',
              help="Type of transformation requested, choose between"
                   + " ['all', 'blur', 'mask', 'roi', 'analysis', "
                   + "'pseudolandmarks', 'colors]")
@click.argument('file', required=False)
def main(file, src, dst, type) -> None:
    if file is not None:
        if os.path.isfile(file) is False:
            return print(f"{file} does not exist or is not a file")
        if (filetype.guess(file) is None
           or filetype.guess(file).extension != 'jpg'):
            return print(f"{file} is not a jpeg image")
        transform_image(file, dst='.', type=type)
        plot_images(file, dst='.')
    elif (src is not None and dst is not None):
        if os.path.isdir(src) is False:
            return print(f"{src} does not exist or is not a directory")
        transform_directory(src=src, dst=dst, type=type)
    else:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        print("You must either provide a [src] and [dst] directory, or a "
              + "source file as arguments for the transformation")
        ctx.exit()


if __name__ == "__main__":
    main()
