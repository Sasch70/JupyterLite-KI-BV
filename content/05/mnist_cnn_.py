# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 00:30:38 2022

@author: stefan.kray
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

model = None
batch_size=128

(ds_validate,ds_test,ds_train), ds_info = tfds.load(
    'mnist',
    split=['train[:10%]','train[10%:20%]','train[20%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_validate = ds_validate.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_validate = ds_validate.batch(batch_size)
ds_validate = ds_validate.cache()
ds_validate = ds_validate.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


l=list(ds_train.as_numpy_iterator())
# l: 469 batches x 2 Einträge (0: Bild, 1: Label) x 128 (batch size)
plt.imshow(l[0][0][5])
print("label:",l[0][1][5])

# --- sequential model hier einfügen ---
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# -----------------------------------------------
if model:
    model.compile(
        #optimizer=tf.keras.optimizers.Adam(0.005),
        optimizer=tf.keras.optimizers.Adadelta(1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    model.fit(
        ds_train,
        epochs=24,
        validation_data=ds_validate,
    )
    
    print("final loss and accuracy on test set:",model.evaluate(ds_test))
    
    if False: #model speichern
        model.save("mnist_cnn_base_model.h5")
    
    if True:
        model2=tf.keras.Model(inputs=model.inputs,outputs=model.layers[1].output)
        feature_map=model2.predict(l[0][0][5:6]) #(1,26,26,32)
        
        # plot all 64 maps in an 8x8 squares
        square = 8
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                ax.imshow(feature_map[0, :, :, ix-1])
                ix += 1
        # show the figure
        plt.subplots_adjust(wspace=-0.8,hspace=0.1)
        plt.show()
        
        w=model.layers[0].get_weights()
        
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(8):
            for _ in range(4):
                # specify subplot and turn of axis
                ax = plt.subplot(4, 8, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                ax.imshow(w[0][:, :, 0, ix-1])
                ix += 1
        # show the figure
        #plt.subplots_adjust(wspace=-0.8,hspace=0.1)
        plt.show()