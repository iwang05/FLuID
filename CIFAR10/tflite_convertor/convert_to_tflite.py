import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

"""Define the base model for CIFAR10.

To be compatible with TFLite Model Personalization, we need to define a
base model and a head model. 

Here we are using an identity layer for base model, which just passes the 
input as it is to the head model.
"""

base = tf.keras.Sequential(
    [tf.keras.Input(shape=(32, 32, 3)), tf.keras.layers.Lambda(lambda x: x)]
)
base.compile(loss="categorical_crossentropy", optimizer="adam")
base.save("identity_model", save_format="tf")

"""Define the head model.

This is the model architecture that we will train using Flower. 
"""
# set sub-model size (p)
p=0.95
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(int(p*32), (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(tf.keras.layers.Conv2D(int(p*32), (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(int(p*64), (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(int(p*64), (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(int(p*128), (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(int(p*128), (3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(int(p*512), activation='relu'))
model.add(tf.keras.layers.Dense(int(p*256), activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))    # num_classes = 10

model.build()
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()



"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""

base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    10, base_path, heads.KerasModelHead(model), optimizers.Adam(1e-4), train_batch_size=32
)

converter.convert_and_save("tflite_model")
