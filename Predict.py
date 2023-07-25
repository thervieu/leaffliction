import os, sys
import filetype
import numpy as np

import joblib

from plantcv import plantcv as pcv

from keras.preprocessing import image

from Transformation import transform_gaussian_blur, transform_masked
from Transformation import transform_roi, transform_analysis
from Transformation import transform_pseudolandmarks


def help():
    print(f'usage: python3 Predict.py [image_path]')


def make_images(path, fruit):
    pcv.params.debug_outdir = "."
    img, path, filename = pcv.readimage(path)

    img_b = transform_gaussian_blur(img)
    pcv.print_image(img_b, f"{fruit}_BLURRED.JPG")
    img_p = transform_pseudolandmarks(img)
    pcv.print_image(img_p, f"{fruit}_PSEUDOLANDMARKS.JPG")
    img_m = transform_masked(img)
    pcv.print_image(img_m, f"{fruit}_MASKED.JPG")
    img_roi = transform_roi(img)
    pcv.print_image(img_roi, f"{fruit}_ROI_OBJECTS.JPG")
    img_a = transform_analysis(img)
    pcv.print_image(img_a, f"{fruit}_ANALYZED.JPG")


def remove_images(fruit):
    os.remove(f'./{fruit}_BLURRED.JPG')
    os.remove(f'./{fruit}_PSEUDOLANDMARKS.JPG')
    os.remove(f'./{fruit}_MASKED.JPG')
    os.remove(f'./{fruit}_ROI_OBJECTS.JPG')
    os.remove(f'./{fruit}_ANALYZED.JPG')


def load_image(type, fruit):
    path = None

    if type=='blur':
        path = f'./{fruit}_BLURRED.JPG'
    if type=='pseudolandmarks':
        path = f'./{fruit}_PSEUDOLANDMARKS.JPG'
    if type=='mask':
        path = f'./{fruit}_MASKED.JPG'
    if type=='roi':
        path = f'./{fruit}_ROI_OBJECTS.JPG'
    if type=='analysis':
        path = f'./{fruit}_ANALYZED.JPG'

    img = image.load_img(path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    
    return img_tensor


def soft_vote(predictions):
    # Compute the average prediction for each sample and class (soft vote)
    ensemble_prediction = np.mean(predictions, axis=0)

    # Convert the ensemble predictions to class labels (index of the maximum probability)
    ensemble_prediction = np.argmax(ensemble_prediction)
    
    return ensemble_prediction


def hard_vote(predictions):
    nb_classes = len(predictions[0])

    pred_array = [pred for i in range(len(predictions)) for pred in predictions[i] ]

    # Compute the majority vote for each sample and class (hard vote)
    pred = np.argmax(pred_array)%nb_classes

    return pred


def main():
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} is not a file".format(sys.argv[1]))
    if (filetype.guess(sys.argv[1]) is None
        or filetype.guess(sys.argv[1]).extension != 'jpg'):
        return print(f"{sys.argv[1]} is not a jpeg image")
 
    jl_name = "Apples/Apples.joblib" if "Apples" in sys.argv[1] else "Grapes/Grapes.joblib"
    models = joblib.load(filename="images_all/"+jl_name)

    fruit = "Apple" if "Apples" in sys.argv[1] else "Grape"
    make_images(sys.argv[1], fruit)

    predictions = []
    transformations = ['blur', 'pseudolandmarks', 'mask', 'roi', 'analysis']
    for i in range(len(models)):
        prediction = models[i].predict(load_image(transformations[i], fruit))
        predictions.append(prediction[0])

    remove_images(fruit)

    s_vote = soft_vote(predictions)
    h_vote = hard_vote(predictions)
    classes = sorted(os.listdir(os.path.dirname(os.path.dirname(sys.argv[1]))))
    print(f'soft voting predicted : {classes[s_vote]}')
    print(f'hard voting predicted : {classes[h_vote]}')


if __name__ == "__main__":
    main()