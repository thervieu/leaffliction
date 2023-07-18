import os, sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import image_dataset_from_directory


def help():
    print(f'usage: python3 P4.py [folder with images]')


def make_model(dataset):
    model = models.Sequential()
    model.add(layers.Rescaling(1.0 / 255))
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(128, 128, 1),
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(len(dataset.class_names), activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #
        metrics=["accuracy"],
    )
    return model


def main():
    # argument
    if len(sys.argv) != 2:
        return help()
    if os.path.isdir(sys.argv[1]) is False:
        return print("Argument {} is not a directory".format(sys.argv[1]))
    
    #data preprocessing
    data = image_dataset_from_directory(
        sys.argv[1],
        validation_split=0.3, # 0.7 for training, 0.1 for validation, 0.2 for testing
        subset="both",
        seed=42,
        image_size=(128, 128), # takes 4 times less memory and time than (256,256)
    )
    val_batches = tf.data.experimental.cardinality(data[1])
    train_data = data[0]
    validation_data = data[1].skip((2*val_batches) // 3)
    test_data = data[1].take((2*val_batches) // 3)
    
    # create model
    model = make_model(train_data)

    # Create a learning rate scheduler callback.
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=5
    )
    # Create an early stopping callback.
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # fit model
    history = model.fit(
        train_data,
        epochs=5,
        validation_data=validation_data,
        callbacks=[early_stopping, reduce_lr],
    )

    test_loss, test_acc = model.evaluate(test_data, verbose=2)
    print("test_acc : ", test_acc)

    # model.save("model/mymodel.model")



if __name__ == "__main__":
    main()