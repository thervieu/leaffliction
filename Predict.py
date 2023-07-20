import os, sys
import filetype
import numpy as np

from plantcv import plantcv as pcv

from keras.models import load_model
from keras.preprocessing import image


def help():
    print(f'usage: python3 Predict.py [image_path]')



def main():
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isfile(sys.argv[1]) is False:
        return print("Argument {} is not a file".format(sys.argv[1]))
    if (filetype.guess(sys.argv[1]) is None
        or filetype.guess(sys.argv[1]).extension != 'jpg'):
        return print(f"{sys.argv[1]} is not a jpeg image")
    
    model = load_model(filepath="model.keras", compile=True)

    # img, path, filename = pcv.readimage(sys.argv[1])
    # img = pcv.gaussian_blur(img, ksize=(3, 3))
    # img = image.smart_resize(img, (128, 128))
    img = image.load_img(sys.argv[1], target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    
    prediction = model.predict(img_tensor)
    print(f'prediction: {prediction}')

    classes = sorted(os.listdir(os.path.dirname(os.path.dirname(sys.argv[1]))))
    print(f'class predicted: {classes[np.argmax(prediction[0])]}')

    # total_len = 0
    # total_true = 0
    # for class_ in classes:
    #     path = os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[1]))) + "/" + class_
    #     true = 0
    #     listdir = os.listdir(path)
    #     print(class_, len(listdir))
    #     for file in listdir:
    #         img = image.load_img(path + "/" + file, target_size=(128, 128))
    #         img_tensor = image.img_to_array(img)                    # (height, width, channels)
    #         img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #         prediction = model.predict(img_tensor, verbose=0)
    #         if class_ == classes[np.argmax(prediction[0])]:
    #             # print(f'{class_} {classes[np.argmax(prediction[0])]}')
    #             true += 1
    #         if listdir.index(file)==len(listdir)-1:
    #             print(f'Accuracy : {float(true)/float(len(listdir))}')
    #     total_true += true
    #     total_len += len(listdir)
        
    # print(f'Total true : {total_true}')
    # print(f'Total files : {total_len}')
    # print(f'Total Accuracy : {float(total_true)/float(total_len)}')

if __name__ == "__main__":
    main()